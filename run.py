"""Command-line experiments for Euler--Maruyama versus SGLD.

Examples
--------
Quick smoke test::

    python run.py smoke

Sobol sample of the (m,s) plane::

    python run.py sample --epsilons 1e-3 5e-4 --n-points 500 --sampling sobol

Fixed grid::

    python run.py grid --m-list 100 300 1000 --s-list 20 50 100 --epsilons 1e-3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Force CPU only when explicitly requested.  For --device gpu we leave JAX's
# normal backend discovery untouched and validate the result after import.
def _requested_device() -> str | None:
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].lower()
    return None


_REQUESTED_DEVICE = _requested_device()
if _REQUESTED_DEVICE == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cost_funcs import OBSERVABLES, costs, f_ou


FIELDS = (
    "point_idx",
    "m",
    "s",
    "epsilon",
    "euler_cost",
    "sgld_cost",
    "euler_rmse",
    "sgld_rmse",
    "winner",
    "elapsed",
)


def detect_backend(requested: str | None) -> str:
    backend = jax.default_backend()
    devices = jax.devices()
    if requested == "gpu" and backend != "gpu":
        raise RuntimeError(
            "--device gpu was requested, but JAX did not detect a GPU. "
            "Install an appropriate jax[cuda12] build (typically under Linux/WSL2) "
            "or omit --device/choose --device cpu."
        )
    if backend == "gpu":
        print(f"Backend: GPU ({devices[0].device_kind})")
        return "gpu"
    print(f"Backend: {backend.upper()} ({len(devices)} JAX device(s))")
    return backend


def run_one_point(
    m: int,
    s: int,
    epsilon: float,
    *,
    sig_th: float,
    sig_y: float,
    initial: float,
    func_name: str,
    n_experiments: int,
    n_rep: int,
    seed: int,
    with_replacement: bool,
    horizon_factor: float,
    max_validation_paths: int,
) -> dict:
    func = OBSERVABLES[func_name]
    key = jax.random.PRNGKey(seed)
    start = time.perf_counter()
    e_rmse, e_cost, s_rmse, s_cost = costs(
        key,
        int(m),
        int(s),
        float(epsilon),
        float(sig_th),
        float(sig_y),
        func,
        f=f_ou,
        g=1.0 / math.sqrt(m),
        initial=float(initial),
        n_experiments=int(n_experiments),
        n=int(n_rep),
        replace=bool(with_replacement),
        horizon_factor=float(horizon_factor),
        max_validation_paths=int(max_validation_paths),
    )
    elapsed = time.perf_counter() - start
    return {
        "m": int(m),
        "s": int(s),
        "epsilon": float(epsilon),
        "euler_cost": float(e_cost),
        "sgld_cost": float(s_cost),
        "euler_rmse": float(e_rmse),
        "sgld_rmse": float(s_rmse),
        "winner": "Euler" if e_cost <= s_cost else "SGLD",
        "elapsed": float(elapsed),
    }


def _worker(payload: tuple) -> tuple[int, dict]:
    idx, kwargs = payload
    return idx, run_one_point(**kwargs)


def canonical_config(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_id(config: dict) -> str:
    return hashlib.sha256(canonical_config(config).encode("utf-8")).hexdigest()[:10]


def load_results(path: Path, expected_config: dict | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    found_config = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# config: "):
                found_config = json.loads(line[len("# config: "):])
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != len(FIELDS):
                raise ValueError(f"Malformed result row in {path}: {line}")
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
    if expected_config is not None:
        if found_config is None:
            raise ValueError(
                f"{path} has no configuration header and cannot be safely resumed. "
                "Move/delete it or choose another --out-dir."
            )
        if canonical_config(found_config) != canonical_config(expected_config):
            raise ValueError(
                f"Configuration mismatch for existing result file {path}. "
                "Refusing to mix results from different experiments."
            )
    return rows


def append_result(path: Path, config: dict, point_idx: int, result: dict) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write("# config: " + canonical_config(config) + "\n")
            handle.write("# " + "  ".join(FIELDS) + "\n")
        handle.write(
            f"{point_idx:d}  {result['m']:d}  {result['s']:d}  "
            f"{result['epsilon']:.12e}  {result['euler_cost']:.12e}  "
            f"{result['sgld_cost']:.12e}  {result['euler_rmse']:.12e}  "
            f"{result['sgld_rmse']:.12e}  {result['winner']}  "
            f"{result['elapsed']:.6f}\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def save_npz(path: Path, rows: list[dict], config: dict) -> None:
    rows = sorted(rows, key=lambda r: r["point_idx"])
    np.savez(
        path,
        point_idx=np.asarray([r["point_idx"] for r in rows], dtype=int),
        m=np.asarray([r["m"] for r in rows], dtype=int),
        s=np.asarray([r["s"] for r in rows], dtype=int),
        epsilon=np.asarray([r["epsilon"] for r in rows], dtype=float),
        euler_cost=np.asarray([r["euler_cost"] for r in rows], dtype=float),
        sgld_cost=np.asarray([r["sgld_cost"] for r in rows], dtype=float),
        euler_rmse=np.asarray([r["euler_rmse"] for r in rows], dtype=float),
        sgld_rmse=np.asarray([r["sgld_rmse"] for r in rows], dtype=float),
        winner=np.asarray([r["winner"] for r in rows]),
        elapsed=np.asarray([r["elapsed"] for r in rows], dtype=float),
        config_json=np.asarray([canonical_config(config)]),
    )


def common_seed(base_seed: int, m: int, epsilon: float, point_index: int = 0) -> int:
    eps_bits = int(np.float64(epsilon).view(np.uint64))
    sequence = np.random.SeedSequence([
        int(base_seed),
        int(m),
        int(point_index),
        eps_bits & 0xFFFFFFFF,
        (eps_bits >> 32) & 0xFFFFFFFF,
    ])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def sample_ms_uniform(n_points: int, m_min: int, m_max: int, seed: int):
    rng = np.random.default_rng(seed)
    m = rng.integers(m_min, m_max + 1, size=n_points)
    # high is exclusive; using mi+1 includes the full-batch case s=m.
    s = np.asarray([rng.integers(1, int(mi) + 1) for mi in m], dtype=int)
    return m, s


def sample_ms_sobol(n_points: int, m_min: int, m_max: int, seed: int):
    from scipy.stats.qmc import Sobol

    k = int(np.ceil(np.log2(max(n_points, 2))))
    points = Sobol(d=2, scramble=True, seed=seed).random_base2(k)[:n_points]
    m = np.floor(m_min + points[:, 0] * (m_max - m_min + 1)).astype(int)
    m = np.clip(m, m_min, m_max)
    s = np.floor(1 + points[:, 1] * m).astype(int)
    s = np.clip(s, 1, m)
    return m, s


def theoretical_em_boundary(m: np.ndarray, epsilon: float) -> np.ndarray:
    """Boundary s = m - epsilon*m^2 in the m<epsilon^{-1} regime."""
    return m - epsilon * m**2


def plot_winner_map(rows: list[dict], epsilon: float, output: Path, log_scale: bool) -> None:
    if not rows:
        return
    m = np.asarray([r["m"] for r in rows], dtype=float)
    s = np.asarray([r["s"] for r in rows], dtype=float)
    em = np.asarray([r["winner"] == "Euler" for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(m[em], s[em], marker="^", s=32, label="EM cheaper", alpha=0.8)
    ax.scatter(m[~em], s[~em], marker="o", s=28, label="SGLD cheaper", alpha=0.8)

    if log_scale:
        lo = max(1.0, float(np.min(m)))
        curve_m = np.logspace(math.log10(lo), math.log10(float(np.max(m))), 800)
    else:
        curve_m = np.linspace(1.0, float(np.max(m)), 1200)
    boundary = theoretical_em_boundary(curve_m, epsilon)
    valid = (curve_m < 1.0 / epsilon) & (boundary >= 1.0)
    ax.plot(curve_m[valid], boundary[valid], "k--", lw=1.5,
            label=r"$s=m-\varepsilon m^2$")
    ax.plot(curve_m, curve_m, ":", lw=1.2, label=r"$s=m$")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(r"data size $m$")
    ax.set_ylabel(r"mini-batch size $s$")
    ax.set_title(rf"Empirical cost comparison, $\varepsilon={epsilon:g}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def execute_cases(cases: list[tuple[int, int, float]], args, config: dict, txt_path: Path) -> list[dict]:
    previous = load_results(txt_path, expected_config=config) if txt_path.exists() else []
    completed = {r["point_idx"] for r in previous}
    pending = [i for i in range(len(cases)) if i not in completed]
    if not pending:
        print(f"All {len(cases)} points already exist in {txt_path}.")
        return previous

    work = []
    for i in pending:
        m, s, eps = cases[i]
        kwargs = {
            "m": m,
            "s": s,
            "epsilon": eps,
            "sig_th": args.sig_th,
            "sig_y": args.sig_y,
            "initial": args.initial,
            "func_name": args.func,
            "n_experiments": args.n_experiments,
            "n_rep": args.n_rep,
            "seed": common_seed(args.seed, m, eps, i),
            "with_replacement": args.with_replacement,
            "horizon_factor": args.horizon_factor,
            "max_validation_paths": args.max_validation_paths,
        }
        work.append((i, kwargs))

    def record(done: int, idx: int, result: dict):
        append_result(txt_path, config, idx, result)
        print(
            f"[{done}/{len(work)}] m={result['m']}, s={result['s']}, "
            f"eps={result['epsilon']:.2e}: "
            f"EM={result['euler_cost']:.3e}, SGLD={result['sgld_cost']:.3e} "
            f"-> {result['winner']} ({result['elapsed']:.2f}s)"
        )

    if args.workers == 1:
        for done, payload in enumerate(work, start=1):
            idx, result = _worker(payload)
            record(done, idx, result)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            future_map = {pool.submit(_worker, item): item[0] for item in work}
            for done, future in enumerate(as_completed(future_map), start=1):
                idx, result = future.result()
                record(done, idx, result)

    return load_results(txt_path, expected_config=config)


def base_config(args, mode: str) -> dict:
    return {
        "mode": mode,
        "sig_th": args.sig_th,
        "sig_y": args.sig_y,
        "initial": args.initial,
        "func": args.func,
        "n_experiments": args.n_experiments,
        "n_rep": args.n_rep,
        "seed": args.seed,
        "with_replacement": args.with_replacement,
        "horizon_factor": args.horizon_factor,
        "max_validation_paths": args.max_validation_paths,
    }


def run_sample(args, backend: str) -> None:
    sampler = sample_ms_sobol if args.sampling == "sobol" else sample_ms_uniform
    out_dir = Path(args.out_dir)
    for eps in args.epsilons:
        m_values, s_values = sampler(args.n_points, args.m_min, args.m_max, args.seed)
        cases = [(int(m), int(s), float(eps)) for m, s in zip(m_values, s_values)]
        config = base_config(args, "sample") | {
            "epsilon": float(eps),
            "n_points": args.n_points,
            "sampling": args.sampling,
            "m_min": args.m_min,
            "m_max": args.m_max,
            "backend": backend,
        }
        tag = f"sample_{args.func}_eps{eps:.0e}_{config_id(config)}"
        txt_path = out_dir / f"{tag}.txt"
        npz_path = out_dir / f"{tag}.npz"

        if args.plot_only:
            rows = load_results(txt_path, expected_config=config)
        else:
            rows = execute_cases(cases, args, config, txt_path)
            save_npz(npz_path, rows, config)

        plot_winner_map(rows, eps, out_dir / f"{tag}_log.png", log_scale=True)
        plot_winner_map(rows, eps, out_dir / f"{tag}_linear.png", log_scale=False)


def run_grid(args, backend: str) -> None:
    cases: list[tuple[int, int, float]] = []
    for m in args.m_list:
        for s in args.s_list:
            if not (1 <= s <= m):
                print(f"Skipping invalid pair m={m}, s={s}; require 1 <= s <= m.")
                continue
            for eps in args.epsilons:
                cases.append((int(m), int(s), float(eps)))
    if not cases:
        raise ValueError("No valid grid cases were supplied.")

    config = base_config(args, "grid") | {
        "m_list": args.m_list,
        "s_list": args.s_list,
        "epsilons": args.epsilons,
        "backend": backend,
    }
    tag = f"grid_{args.func}_{config_id(config)}"
    out_dir = Path(args.out_dir)
    txt_path = out_dir / f"{tag}.txt"
    rows = execute_cases(cases, args, config, txt_path)
    save_npz(out_dir / f"{tag}.npz", rows, config)


def run_smoke(args, backend: str) -> None:
    result = run_one_point(
        8,
        2,
        0.2,
        sig_th=args.sig_th,
        sig_y=args.sig_y,
        initial=args.initial,
        func_name=args.func,
        n_experiments=8,
        n_rep=1,
        seed=args.seed,
        with_replacement=args.with_replacement,
        horizon_factor=0.5,
        max_validation_paths=32,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    for key in ("euler_cost", "sgld_cost", "euler_rmse", "sgld_rmse"):
        if not math.isfinite(result[key]):
            raise RuntimeError(f"Smoke test produced non-finite {key}.")
    print("Smoke test passed.")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sig-th", type=float, default=0.2,
                        help="sigma_theta in theta~N(0,sigma_theta^2/m) (default 0.2)")
    parser.add_argument("--sig-y", type=float, default=0.15,
                        help="observation-noise standard deviation (default 0.15)")
    parser.add_argument("--initial", type=float, default=1.0,
                        help="initial state of each Langevin path (default 1.0)")
    parser.add_argument("--func", choices=sorted(OBSERVABLES), default="sin",
                        help="observable applied to the terminal state")
    parser.add_argument("--n-experiments", type=int, default=50,
                        help="pilot paths for step-doubling and variance estimation")
    parser.add_argument("--n-rep", type=int, default=5,
                        help="independently regenerated datasets per point")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel point workers; use 1 on GPU")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--with-replacement", action="store_true",
                        help="use with-replacement mini-batches (paper default is without replacement)")
    parser.add_argument("--horizon-factor", type=float, default=3.0,
                        help="T=ceil(factor*log(1/epsilon)); paper default is 3")
    parser.add_argument("--max-validation-paths", type=int, default=4096,
                        help="cap paths used only for the RMSE diagnostic; 0 means no cap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Empirical operation-count comparison of EM and SGLD."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    smoke = sub.add_parser("smoke", help="fast end-to-end sanity check")
    add_common_args(smoke)

    sample = sub.add_parser("sample", help="sample (m,s) pairs and draw winner maps")
    sample.add_argument("--epsilons", nargs="+", type=float, default=[1e-3, 5e-4])
    sample.add_argument("--n-points", type=int, default=500)
    sample.add_argument("--sampling", choices=("uniform", "sobol"), default="sobol")
    sample.add_argument("--m-min", type=int, default=2)
    sample.add_argument("--m-max", type=int, default=1000)
    sample.add_argument("--plot-only", action="store_true")
    add_common_args(sample)

    grid = sub.add_parser("grid", help="evaluate a user-specified parameter grid")
    grid.add_argument("--m-list", nargs="+", type=int, required=True)
    grid.add_argument("--s-list", nargs="+", type=int, required=True)
    grid.add_argument("--epsilons", nargs="+", type=float, required=True)
    add_common_args(grid)
    return parser


def validate_args(args, backend: str) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if backend == "gpu" and args.workers != 1:
        raise ValueError("Use --workers 1 on GPU to avoid competing JAX processes.")
    if args.n_experiments < 2 or args.n_rep < 1:
        raise ValueError("Need --n-experiments >= 2 and --n-rep >= 1.")
    if args.sig_th <= 0 or args.sig_y <= 0:
        raise ValueError("--sig-th and --sig-y must be positive.")
    if args.max_validation_paths < 0:
        raise ValueError("--max-validation-paths must be non-negative.")
    if args.mode == "sample":
        if args.n_points < 1:
            raise ValueError("--n-points must be positive.")
        if args.m_min < 2 or args.m_max < args.m_min:
            raise ValueError("Require 2 <= --m-min <= --m-max.")
    if hasattr(args, "epsilons") and any(not (0 < e < 1) for e in args.epsilons):
        raise ValueError("Every epsilon must lie in (0,1).")


def main() -> None:
    args = build_parser().parse_args()
    backend = detect_backend(args.device)
    validate_args(args, backend)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")

    if args.mode == "smoke":
        run_smoke(args, backend)
    elif args.mode == "sample":
        run_sample(args, backend)
    else:
        run_grid(args, backend)


if __name__ == "__main__":
    main()
