"""
Fast empirical cost comparison: Euler vs SGLD.

Single entry point for both experiment modes:
  - `python run.py sample ...`  Random/quasi-random (m, s) sampling over a
    region, one plot per epsilon (this was run_fast.py).
  - `python run.py grid ...`    Every (m, s, epsilon) combination from
    user-supplied lists (this was run_fast_table.py).

Both modes share the JIT-compiled / lax.scan backend in
src/cost_funcs.py, incremental/resumable file output, optional
multi-process evaluation via --workers, and --device cpu|gpu selection.

Variable names follow the original scripts.
"""

import argparse
import math
import os
import sys
import time as _time

# ---------------------------------------------------------------------------
# Device selection MUST happen before any other JAX import triggers init.
# We parse --device early and configure JAX accordingly.
# ---------------------------------------------------------------------------
def _early_device_parse():
    """Extract --device from sys.argv before full argparse runs."""
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].lower()
    return None


_DEVICE = _early_device_parse()
if _DEVICE is not None:
    _JAX_PLATFORM = "cuda" if _DEVICE == "gpu" else _DEVICE
    os.environ["JAX_PLATFORMS"] = _JAX_PLATFORM

from concurrent.futures import ProcessPoolExecutor, as_completed

import jax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ---------------------------------------------------------------------------
# Make `src` importable regardless of the current working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.cost_funcs import costs, f_ou as drift


# ========================== model parameters ================================
# sig_th, sig_y, initial, and func are user-configurable via CLI flags (see
# _add_common_args). FUNC_REGISTRY maps the --func name to the actual
# functional applied to the terminal state, func(y). It is looked up by name
# inside _worker (rather than passed as a closure) so it resolves correctly
# in subprocesses spawned by ProcessPoolExecutor.
FUNC_REGISTRY = {
    "square": lambda y: y**2,
    "sin": lambda y: jax.numpy.sin(y),
    "sin_abs": lambda y: jax.numpy.sin(y) + jax.numpy.abs(y) + 1,
}


def _detect_backend():
    """Detect JAX backend, print status, and return tag string.

    If --device gpu was requested but no GPU is available, falls back to CPU
    and prints a recommendation to increase --workers.
    """
    try:
        platform = jax.devices()[0].platform
    except Exception:
        platform = "cpu"

    tag = "gpu" if platform == "cuda" else platform
    requested = _DEVICE

    if requested == "gpu" and tag != "gpu":
        print("WARNING: GPU was requested but is not available. "
              "Falling back to CPU.")
        print("  --> Consider increasing --workers (e.g. --workers 12) "
              "to compensate.")
        print()

    if tag == "gpu":
        dev_name = jax.devices()[0].device_kind
        print(f"Backend: GPU ({dev_name})")
    else:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        print(f"Backend: CPU ({n_cores} logical cores)")
        if requested != "gpu":
            print("  --> For GPU acceleration, run via WSL2 with jax[cuda12].")
    return tag


# ========================== single-point evaluation ========================
def run_one_point(m, s, epsilon,
                  sig_th, sig_y,
                  func, drift, initial,
                  n_experiments=10**2,
                  n_rep=10,
                  K=None,
                  seed=None,
                  vectorized=False):

    g = 1.0 / math.sqrt(m)

    if K is None:
        K = (sig_th**(-2) + sig_y**(-2)) / 2

    # deterministic key from seed
    key = jax.random.PRNGKey(seed if seed is not None else 0)

    t0 = _time.perf_counter()
    e_rmse, e_cost, s_rmse, s_cost = costs(
        key,
        int(m), int(s), epsilon,
        sig_th, sig_y,
        func, drift, g,
        initial,
        n_experiments=int(n_experiments),
        n=int(n_rep),
        K=K,
        vectorized=vectorized,
    )
    elapsed = _time.perf_counter() - t0

    e_cost_f = float(e_cost)
    s_cost_f = float(s_cost)

    winner = "Euler" if e_cost_f <= s_cost_f else "SGLD"

    return {
        "m": int(m),
        "s": int(s),
        "epsilon": float(epsilon),
        "euler_cost": e_cost_f,
        "sgld_cost": s_cost_f,
        "euler_rmse": float(e_rmse),
        "sgld_rmse": float(s_rmse),
        "winner": winner,
        "elapsed": elapsed,
    }


# ========================== shared worker (subprocess) =====================
def _worker(args):
    """Picklable top-level function for ProcessPoolExecutor."""
    (idx, m, s, epsilon, sig_th, sig_y, initial, func_name,
     n_experiments, n_rep, K, seed, vectorized) = args
    func = FUNC_REGISTRY[func_name]
    res = run_one_point(
        m, s, epsilon,
        sig_th, sig_y,
        func, drift, initial,
        n_experiments=n_experiments,
        n_rep=n_rep,
        K=K,
        seed=seed,
        vectorized=vectorized,
    )
    return idx, res


# ========================== shared row I/O ==================================
# Canonical row schema (one dict per completed point):
#   point_idx, m, s, epsilon, euler_cost, sgld_cost,
#   euler_rmse, sgld_rmse, winner, elapsed
_FIELDS = (
    "point_idx", "m", "s", "epsilon",
    "euler_cost", "sgld_cost", "euler_rmse", "sgld_rmse",
    "winner", "elapsed",
)


def write_result_row(filename, point_idx, epsilon, res):
    """Append one completed point immediately (safe to interrupt/resume)."""
    needs_header = (
        not os.path.exists(filename)
        or os.path.getsize(filename) == 0
    )

    with open(filename, "a", encoding="utf-8") as fh:
        if needs_header:
            fh.write("# " + "  ".join(_FIELDS) + "\n")

        fh.write(
            f"{int(point_idx)}  {res['m']}  {res['s']}  {epsilon:.6e}  "
            f"{res['euler_cost']:.6e}  {res['sgld_cost']:.6e}  "
            f"{res['euler_rmse']:.6e}  {res['sgld_rmse']:.6e}  "
            f"{res['winner']}  {res['elapsed']:.4f}\n"
        )

        # Make the completed row available immediately, even if a later
        # point fails or the run is interrupted.
        fh.flush()
        os.fsync(fh.fileno())


def load_results(path):
    """Load rows written by write_result_row as a list of dicts."""
    rows = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < len(_FIELDS):
                continue

            rows.append({
                "point_idx": int(parts[0]),
                "m": int(parts[1]),
                "s": int(parts[2]),
                "epsilon": float(parts[3]),
                "euler_cost": float(parts[4]),
                "sgld_cost": float(parts[5]),
                "euler_rmse": float(parts[6]),
                "sgld_rmse": float(parts[7]),
                "winner": parts[8],
                "elapsed": float(parts[9]),
            })

    return rows


def save_results_npz(path, rows, total_seconds):
    """Save results as .npz for structured comparison."""
    np.savez(
        path,
        point_idx=np.array([r["point_idx"] for r in rows]),
        m=np.array([r["m"] for r in rows]),
        s=np.array([r["s"] for r in rows]),
        epsilon=np.array([r["epsilon"] for r in rows]),
        euler_cost=np.array([r["euler_cost"] for r in rows]),
        sgld_cost=np.array([r["sgld_cost"] for r in rows]),
        euler_rmse=np.array([r["euler_rmse"] for r in rows]),
        sgld_rmse=np.array([r["sgld_rmse"] for r in rows]),
        winner=np.array([r["winner"] for r in rows]),
        elapsed_seconds=np.array([r["elapsed"] for r in rows]),
        total_seconds=np.array([total_seconds]),
    )
    print(f"  Saved {path}")


def _completed_point_indices(path):
    """Return the set of point_idx values already saved."""
    return {r["point_idx"] for r in load_results(path)}


# ========================== plotting =======================================
def plot_from_rows(rows, epsilon, save_path=None):
    """Log-log scatter of Euler vs SGLD winners."""
    m = np.array([r["m"] for r in rows], dtype=float)
    s = np.array([r["s"] for r in rows], dtype=float)
    is_euler = np.array([r["winner"] == "Euler" for r in rows])

    plt.figure(figsize=(8, 6))

    plt.scatter(
        m[is_euler], s[is_euler],
        color="tab:red", marker="^",
        s=50, edgecolor="k", linewidth=0.3,
        alpha=0.85, label="Euler cheaper",
    )
    plt.scatter(
        m[~is_euler], s[~is_euler],
        color="tab:green", marker="o",
        s=45, edgecolor="k", linewidth=0.3,
        alpha=0.85, label="SGLD cheaper",
    )

    m_curve = np.logspace(0, np.log10(max(m)), 400)
    s_curve = m_curve - epsilon * m_curve**2
    mask = s_curve > 0
    plt.plot(m_curve[mask], s_curve[mask], "k--", linewidth=2,
             label=r"$(m-s)/m^2=\varepsilon$")

    m_ref = np.logspace(0, np.log10(max(m)), 200)
    plt.plot(m_ref, m_ref, color="gray", linestyle=":", linewidth=2, label=r"$m=s$")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, max(m) * 1.05)
    plt.ylim(1, max(s) * 1.05)
    plt.xlabel(r"$m$")
    plt.ylabel(r"$s$")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"  Saved {save_path}")
    plt.close()


def plot_from_rows_linear(rows, epsilon, title=None, save_path=None):
    """Linear-scale scatter with zoomed inset."""
    m = np.array([r["m"] for r in rows], dtype=float)
    s = np.array([r["s"] for r in rows], dtype=float)
    is_euler = np.array([r["winner"] == "Euler" for r in rows])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(m[is_euler], s[is_euler],
               color="tab:red", marker="^", s=40, alpha=0.8,
               label="Euler cheaper", zorder=3)
    ax.scatter(m[~is_euler], s[~is_euler],
               color="tab:green", marker="o", s=35, alpha=0.8,
               label="SGLD cheaper", zorder=2)

    m_ref = np.linspace(0, max(m), 800)
    ax.plot(m_ref, m_ref, ":", color="gray", lw=2, label=r"$m=s$", zorder=1)

    m_curve = np.linspace(0, max(m), 4000)
    s_curve = m_curve - epsilon * m_curve**2
    mask = s_curve >= 0
    ax.plot(m_curve[mask], s_curve[mask], "k--", lw=3,
            label=r"$(m-s)/m^2=\varepsilon$", zorder=10)

    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$s$")
    ax.set_xlim(0, max(m) * 1.05)
    ax.set_ylim(0, max(s) * 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title(title or f"Linear-scale view (epsilon={epsilon})")

    plt.tight_layout()

    axins = inset_axes(
        ax,
        width="100%", height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.05, 0.38, 0.38),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    axins.scatter(m[is_euler], s[is_euler],
                  color="tab:red", marker="^", s=25, alpha=0.8)
    axins.scatter(m[~is_euler], s[~is_euler],
                  color="tab:green", marker="o", s=22, alpha=0.8)
    axins.plot(m_ref, m_ref, ":", color="gray", lw=1.5)
    axins.plot(m_curve[mask], s_curve[mask], "k--", lw=2.5)
    axins.set_xlim(0, 15)
    axins.set_ylim(0, 20)
    axins.grid(True, linestyle="--", alpha=0.3)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35", lw=1.5)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"  Saved {save_path}")
    plt.close()


# ========================== "sample" mode ===================================
# Random / quasi-random (m, s) sampling over a region; one output file and
# plot pair per epsilon value.

def sample_ms_uniform(n_points, m_min=2, m_max=10**2, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.integers(m_min, m_max + 1, size=n_points)
    s = np.array([rng.integers(1, mi - 1) for mi in m], dtype=int)
    return m, s


def sample_ms_sobol(n_points, m_min=2, m_max=10**2, seed=0):
    """Quasi-random (m, s) pairs via Sobol sequence for better coverage.

    Generates 2D Sobol points in [0,1]^2 and maps them to the triangular
    region where m in [m_min, m_max] and s in [1, m-1].
    """
    from scipy.stats.qmc import Sobol

    sampler = Sobol(d=2, scramble=True, seed=seed)
    # Sobol requires n = 2^k; generate enough then truncate
    k = int(np.ceil(np.log2(max(n_points, 2))))
    n_raw = 2**k
    pts = sampler.random(n_raw)[:n_points]  # shape (n_points, 2)

    u1, u2 = pts[:, 0], pts[:, 1]

    # Map u1 -> m in [m_min, m_max]
    m = np.floor(m_min + u1 * (m_max - m_min + 1)).astype(int)
    m = np.clip(m, m_min, m_max)

    # Map u2 -> s in [1, m-1], conditional on m
    s = np.floor(1 + u2 * (m - 1)).astype(int)
    s = np.clip(s, 1, m - 1)

    return m, s


def run_empirical_map(n_points,
                      epsilon,
                      sig_th, sig_y,
                      func_name, initial,
                      n_experiments=10**2,
                      n_rep=10,
                      K=None,
                      m_min=2,
                      m_max=10**2,
                      seed=0,
                      save_txt="results.txt",
                      n_workers=1,
                      vectorized=False,
                      sampling="uniform"):

    _sampler = sample_ms_sobol if sampling == "sobol" else sample_ms_uniform
    m_list, s_list = _sampler(
        n_points=n_points, m_min=m_min, m_max=m_max, seed=seed
    )

    # Resume by explicit point index rather than by row count. This is
    # required because parallel futures finish out of order.
    completed_points = {
        i for i in _completed_point_indices(save_txt)
        if 0 <= i < n_points
    }

    if len(completed_points) >= n_points:
        print(
            f"  {save_txt} already has {len(completed_points)}/{n_points} "
            "results — skipping."
        )
        return load_results(save_txt)

    pending_indices = [
        i for i in range(n_points)
        if i not in completed_points
    ]
    remaining = len(pending_indices)

    if completed_points:
        print(
            f"  Resuming: {len(completed_points)}/{n_points} points already "
            f"saved; {remaining} points remain."
        )

    # Deterministic unique seed for every original point index.
    rng = np.random.default_rng(seed)
    all_seeds = rng.integers(0, 2**31, size=n_points)

    work = [
        (
            idx,
            int(m_list[idx]),
            int(s_list[idx]),
            epsilon,
            sig_th,
            sig_y,
            initial,
            func_name,
            n_experiments,
            n_rep,
            K,
            int(all_seeds[idx]),
            vectorized,
        )
        for idx in pending_indices
    ]

    n_success = 0
    n_failed = 0

    def _handle(n_finished, point_idx, m, s, idx=None, res=None, exc=None):
        nonlocal n_success, n_failed
        if exc is not None:
            n_failed += 1
            print(
                f"  [{n_finished}/{remaining}] FAILED "
                f"(point {point_idx + 1}) m={m}, s={s}, "
                f"epsilon={epsilon}: {exc}"
            )
            return

        write_result_row(save_txt, idx, epsilon, res)
        n_success += 1
        print(
            f"  [{n_finished}/{remaining}] (point {idx + 1}) "
            f"m={res['m']:4d}, s={res['s']:4d} | "
            f"cost(E)={res['euler_cost']:.3e}, "
            f"cost(S)={res['sgld_cost']:.3e} "
            f"-> {res['winner']}  ({res['elapsed']:.2f}s)"
        )

    if n_workers <= 1:
        for n_finished, item in enumerate(work, start=1):
            point_idx, m, s = item[0], item[1], item[2]
            try:
                idx, res = _worker(item)
            except Exception as exc:
                _handle(n_finished, point_idx, m, s, exc=exc)
                continue
            _handle(n_finished, point_idx, m, s, idx=idx, res=res)
    else:
        # Parallel workers compute only; the main process performs all writes.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, item): item for item in work}

            for n_finished, fut in enumerate(as_completed(futures), start=1):
                item = futures[fut]
                point_idx, m, s = item[0], item[1], item[2]
                try:
                    idx, res = fut.result()
                except Exception as exc:
                    _handle(n_finished, point_idx, m, s, exc=exc)
                    continue
                _handle(n_finished, point_idx, m, s, idx=idx, res=res)

    total_saved = len(_completed_point_indices(save_txt))
    print(
        f"  Saved this run: {n_success}; failed this run: {n_failed}; "
        f"total saved: {total_saved}/{n_points}."
    )

    if total_saved < n_points:
        print(
            "  Re-run the same command to retry only the missing/failed "
            "points."
        )

    return load_results(save_txt)


def run_sample_mode(args, dev):
    for eps in args.epsilons:
        eps_tag = f"{eps:.0e}"
        txt_path = os.path.join(
            args.out_dir, f"Mixed_empirical_results_eps{eps_tag}_{dev}.txt"
        )
        npz_path = os.path.join(
            args.out_dir, f"Mixed_empirical_results_eps{eps_tag}_{dev}.npz"
        )
        loglog_path = os.path.join(
            args.out_dir, f"Empirical Plot (epsilon={eps})_{dev}.png"
        )
        linear_path = os.path.join(
            args.out_dir, f"Empirical Linear-scale view (epsilon={eps})_{dev}.png"
        )

        if not args.plot_only:
            print(f"\n{'='*60}")
            print(f"epsilon = {eps}  |  device = {dev}")
            print(f"{'='*60}")
            t_start = _time.perf_counter()
            rows = run_empirical_map(
                n_points=args.n_points,
                epsilon=eps,
                sig_th=args.sig_th, sig_y=args.sig_y,
                func_name=args.func, initial=args.initial,
                n_experiments=args.n_experiments,
                n_rep=args.n_rep,
                m_min=args.m_min,
                m_max=args.m_max,
                seed=args.seed,
                save_txt=txt_path,
                n_workers=args.workers,
                vectorized=args.vectorized,
                sampling=args.sampling,
            )
            total_sec = _time.perf_counter() - t_start
            print(f"  Total wall-clock: {total_sec:.1f}s")
            save_results_npz(npz_path, rows, total_sec)
        else:
            rows = None

        if os.path.exists(txt_path):
            print(f"\nGenerating plots for epsilon={eps} …")
            if rows is None:
                rows = load_results(txt_path)
            if rows:
                plot_from_rows(rows, epsilon=eps, save_path=loglog_path)
                plot_from_rows_linear(rows, epsilon=eps, save_path=linear_path)
        else:
            print(f"  No results file found at {txt_path}, skipping plots.")


# ========================== "grid" mode =====================================
# Every (m, s, epsilon) combination from user-supplied lists, in one file.

def build_cases(m_list, s_list, eps_list):
    """
    Build all valid (m, s, epsilon) combinations from user-provided lists.

    Returns
    -------
    cases : list of tuples
        [(m, s, epsilon), ...]
    """
    cases = []

    for m in m_list:
        if m < 2:
            print(f"Skipping invalid m={m} (need m >= 2).")
            continue

        for s in s_list:
            if not (1 <= s < m):
                print(f"Skipping invalid pair (m={m}, s={s}) because need 1 <= s < m.")
                continue

            for eps in eps_list:
                if eps <= 0:
                    print(f"Skipping invalid epsilon={eps} (need epsilon > 0).")
                    continue
                cases.append((int(m), int(s), float(eps)))

    return cases


def make_common_seed(base_seed, m, epsilon):
    """Return a deterministic seed shared by all s for the same (m, epsilon).

    This makes Euler use the same random data and Monte Carlo randomness when
    only the minibatch size s changes. The seed remains different for
    different m or epsilon values.
    """
    epsilon_bits = int(np.float64(epsilon).view(np.uint64))

    seed_sequence = np.random.SeedSequence([
        int(base_seed),
        int(m),
        epsilon_bits & 0xFFFFFFFF,
        (epsilon_bits >> 32) & 0xFFFFFFFF,
    ])

    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def run_fixed_cases(cases,
                    sig_th, sig_y,
                    func_name, initial,
                    n_experiments=10**2,
                    n_rep=10,
                    K=None,
                    seed=0,
                    save_txt="results.txt",
                    n_workers=1,
                    vectorized=False):

    # Resume by explicit case index (index into `cases`), same convention as
    # "sample" mode so out-of-order parallel completion is safe.
    completed = {
        i for i in _completed_point_indices(save_txt)
        if 0 <= i < len(cases)
    }

    if len(completed) >= len(cases):
        print(f"  {save_txt} already has {len(completed)}/{len(cases)} results — skipping.")
        return load_results(save_txt)

    pending_indices = [i for i in range(len(cases)) if i not in completed]
    remaining = len(pending_indices)

    if completed:
        print(f"  Resuming: {len(completed)}/{len(cases)} cases already saved; "
              f"{remaining} remain.")

    work = [
        (
            i,
            *cases[i],
            sig_th,
            sig_y,
            initial,
            func_name,
            n_experiments,
            n_rep,
            K,
            make_common_seed(seed, cases[i][0], cases[i][2]),
            vectorized,
        )
        for i in pending_indices
    ]

    n_success = 0
    n_failed = 0

    def _handle(n_finished, case_idx, m, s, epsilon, idx=None, res=None, exc=None):
        nonlocal n_success, n_failed
        if exc is not None:
            n_failed += 1
            print(
                f"  [{n_finished}/{remaining}] FAILED "
                f"(case {case_idx + 1}) m={m}, s={s}, epsilon={epsilon}: {exc}"
            )
            return

        write_result_row(save_txt, idx, epsilon, res)
        n_success += 1
        print(
            f"  [{n_finished}/{remaining}] (case {idx + 1}/{len(cases)}) "
            f"m={res['m']:6d}, s={res['s']:6d}, eps={epsilon:.1e} | "
            f"cost(E)={res['euler_cost']:.3e}, cost(S)={res['sgld_cost']:.3e} "
            f"-> {res['winner']}  ({res['elapsed']:.2f}s)"
        )

    if n_workers <= 1:
        for n_finished, item in enumerate(work, start=1):
            case_idx, m, s, epsilon = item[0], item[1], item[2], item[3]
            try:
                idx, res = _worker(item)
            except Exception as exc:
                _handle(n_finished, case_idx, m, s, epsilon, exc=exc)
                continue
            _handle(n_finished, case_idx, m, s, epsilon, idx=idx, res=res)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, item): item for item in work}

            for n_finished, fut in enumerate(as_completed(futures), start=1):
                item = futures[fut]
                case_idx, m, s, epsilon = item[0], item[1], item[2], item[3]
                try:
                    idx, res = fut.result()
                except Exception as exc:
                    _handle(n_finished, case_idx, m, s, epsilon, exc=exc)
                    continue
                _handle(n_finished, case_idx, m, s, epsilon, idx=idx, res=res)

    total_saved = len(_completed_point_indices(save_txt))
    print(f"  Saved this run: {n_success}; failed this run: {n_failed}; "
          f"total saved: {total_saved}/{len(cases)}.")

    if total_saved < len(cases):
        print("  Re-run the same command to retry only the missing/failed cases.")

    return load_results(save_txt)


def run_grid_mode(args, dev):
    cases = build_cases(args.m_list, args.s_list, args.epsilons)

    if not cases:
        print("No valid (m, s, epsilon) cases to run.")
        return

    print(f"  valid cases     {len(cases)}")

    txt_path = os.path.join(args.out_dir, f"fixed_empirical_results_{dev}.txt")
    npz_path = os.path.join(args.out_dir, f"fixed_empirical_results_{dev}.npz")

    print(f"\n{'='*60}")
    print(f"Running user-specified grid | device = {dev}")
    print(f"{'='*60}")

    t_start = _time.perf_counter()
    rows = run_fixed_cases(
        cases=cases,
        sig_th=args.sig_th, sig_y=args.sig_y,
        func_name=args.func, initial=args.initial,
        n_experiments=args.n_experiments,
        n_rep=args.n_rep,
        K=None,
        seed=args.seed,
        save_txt=txt_path,
        n_workers=args.workers,
        vectorized=args.vectorized,
    )
    total_sec = _time.perf_counter() - t_start

    print(f"  Total wall-clock: {total_sec:.1f}s")
    save_results_npz(npz_path, rows, total_sec)


# ========================== CLI entry point ================================
def _add_common_args(parser):
    parser.add_argument("--sig-th", type=float, default=0.5,
        help="Prior std dev of mu, mu ~ N(0, sig_th^2) (default: 0.5).")
    parser.add_argument("--sig-y", type=float, default=0.25,
        help="Observation noise std dev, Y_i = mu + sig_y * eps_i "
             "(default: 0.25).")
    parser.add_argument("--initial", type=float, default=1.0,
        help="Initial state for the Euler/SGLD paths (default: 1.0).")
    parser.add_argument("--func", type=str, default="square",
        choices=sorted(FUNC_REGISTRY),
        help="Functional func(y) applied to the terminal state and to the "
             "reference posterior integral (default: square, i.e. y**2).")
    parser.add_argument("--n-experiments", type=int, default=10**2,
        help="Pilot paths for stepsize search and variance estimate v2 -> "
             "N_full = 2*v2/eps^2 (default: 100).")
    parser.add_argument("--n-rep", type=int, default=10,
        help="Independent repetitions per (m, s) point; RMSE and cost are "
             "averaged over these (default: 10).")
    parser.add_argument("--seed", type=int, default=2026,
        help="Base seed for reproducibility (default: 2026).")
    parser.add_argument("--workers", type=int, default=1,
        help="Parallel subprocesses across points. CPU: use ~half your "
             "cores. GPU: use 1 (default: 1, sequential).")
    parser.add_argument("--out-dir", type=str, default=".",
        help="Directory for output files (default: cwd).")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "gpu"],
        help="Force cpu/gpu or auto-detect (default: auto-detect).")
    parser.add_argument("--vectorized", action="store_true",
        help="Vectorized stepsize search via vmap (GPU only, auto-disabled "
             "on CPU).")


def main():
    parser = argparse.ArgumentParser(
        description="Fast Euler-vs-SGLD empirical cost experiments."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_sample = subparsers.add_parser(
        "sample",
        help="Random/quasi-random (m, s) sampling over a region "
             "(one plot per epsilon).",
    )
    p_sample.add_argument(
        "--epsilons", nargs="+", type=float,
        default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        help="List of epsilon values to run (default: 1e-1 1e-2 1e-3 1e-4 1e-5).",
    )
    p_sample.add_argument("--n-points", type=int, default=1000,
        help="Number of random (m, s) pairs to evaluate (default: 1000).")
    p_sample.add_argument(
        "--sampling", type=str, default="uniform", choices=["uniform", "sobol"],
        help="How (m, s) pairs are drawn: uniform = pseudo-random, "
             "sobol = quasi-random (default: uniform).",
    )
    p_sample.add_argument("--m-min", type=int, default=2,
        help="Minimum data size m (default: 2).")
    p_sample.add_argument("--m-max", type=int, default=10**3,
        help="Maximum data size m (default: 1000).")
    p_sample.add_argument("--plot-only", action="store_true",
        help="Skip computation, only generate plots from existing txt files.")
    _add_common_args(p_sample)

    p_grid = subparsers.add_parser(
        "grid",
        help="Every (m, s, epsilon) combination from user-supplied lists.",
    )
    p_grid.add_argument("--m-list", nargs="+", type=int, required=True,
        help="List of m values, e.g. --m-list 1000 5000 10000.")
    p_grid.add_argument("--s-list", nargs="+", type=int, required=True,
        help="List of s values, e.g. --s-list 10 50 100.")
    p_grid.add_argument("--epsilons", nargs="+", type=float, required=True,
        help="List of epsilon values, e.g. --epsilons 1e-2 1e-3 1e-4.")
    _add_common_args(p_grid)

    args = parser.parse_args()

    dev = _detect_backend()

    # Disable --vectorized on CPU (it is slower due to padding overhead)
    if args.vectorized and dev != "gpu":
        print("NOTE: --vectorized disabled on CPU (slower due to padding).")
        args.vectorized = False

    print()
    print("-" * 60)
    print(f"Run configuration ({args.mode} mode):")
    print("-" * 60)
    for key, value in sorted(vars(args).items()):
        if key == "mode":
            continue
        print(f"  --{key.replace('_', '-'):<14} {value}")
    print("-" * 60)
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "sample":
        run_sample_mode(args, dev)
    else:
        run_grid_mode(args, dev)

    print("\nDone.")


if __name__ == "__main__":
    main()
