"""Numerical backend for the Euler--Maruyama versus SGLD cost experiments.

The implementation follows the model used in the manuscript:

    theta ~ N(0, sigma_theta^2 / m),
    Y_i | theta ~ N(theta, sigma_y^2),

with posterior N(mu_p, sigma_p^2), and the time-rescaled Langevin SDE

    dX_t = -(X_t-mu_p)/(2 m sigma_p^2) dt + m^{-1/2} dW_t.

SGLD replaces the full-data drift by a mini-batch estimator.  The paper uses
sampling *without replacement* within each mini-batch; that is the default
here.  Mini-batches are independent between time steps and paths.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

Array = jax.Array
Observable = Callable[[Array], Array]
Drift = Callable[[Array, tuple[Array, Array, int]], Array]

# Keep JIT allocations bounded.  Larger Monte Carlo estimators are evaluated
# in chunks of this size.
_MAX_BATCH = 4096

# Safety bound for a single coupled coarse/fine simulation.  If this is hit,
# the requested accuracy/parameter combination is too expensive for the
# current numerical protocol and an informative error is raised.
_MAX_FINE_STEPS = 2_000_000


# ---------------------------------------------------------------------------
# Observables used in the paper
# ---------------------------------------------------------------------------
def observable_square(x: Array) -> Array:
    return x**2


def observable_sin(x: Array) -> Array:
    return jnp.sin(x)


def observable_sin_abs(x: Array) -> Array:
    return jnp.sin(x) + jnp.abs(x) + 1.0


OBSERVABLES: dict[str, Observable] = {
    "square": observable_square,
    "sin": observable_sin,
    "sin_abs": observable_sin_abs,
}


# ---------------------------------------------------------------------------
# Posterior and invariant-measure quantities
# ---------------------------------------------------------------------------
def var_mu_comp(Y: Array, sig_th: float, sig_y: float) -> tuple[Array, Array]:
    """Return posterior variance and mean for the *scaled-prior* model.

    The prior is theta ~ N(0, sig_th^2 / m), not N(0, sig_th^2).  Hence

        sigma_p^2 = m^{-1} (sig_th^{-2} + sig_y^{-2})^{-1},
        mu_p      = mean(Y) / (sig_y^2 / sig_th^2 + 1).
    """
    m = Y.shape[0]
    denom = (sig_y**2 / sig_th**2) + 1.0
    mu_post = jnp.mean(Y) / denom
    precision_sum = sig_th**-2 + sig_y**-2
    var_post = 1.0 / (m * precision_sum)
    return var_post, mu_post


def ref_im_sol(mu_post: Array, var_post: Array, func: Observable) -> Array:
    """Accurate Gaussian expectation of ``func`` by standardized quadrature.

    Integrating on a fixed x-grid is inaccurate when m is large because the
    posterior standard deviation is O(m^{-1/2}).  We instead integrate in the
    standardized variable z=(x-mu)/sigma on a fixed [-10,10] grid.  The grid
    therefore resolves the target equally well for small and large m.
    """
    z = jnp.linspace(-10.0, 10.0, 8193)
    std = jnp.sqrt(var_post)
    x = mu_post + std * z
    phi = jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)
    return trapezoid(func(x) * phi, z)


# ---------------------------------------------------------------------------
# Rescaled OU drift and Euler step
# ---------------------------------------------------------------------------
def f_ou(x: Array, args: tuple[Array, Array, int]) -> Array:
    """Drift of the time-rescaled OU benchmark.

    ``args = (mu_post, var_post, m)`` and

        a(x) = -(x-mu_post)/(2*m*var_post).
    """
    mu_post, var_post, m = args
    return -(x - mu_post) / (2.0 * m * var_post)


def euler_onestep(
    x: Array,
    h: float,
    dW: Array,
    f: Drift,
    g: float,
    args_f: tuple[Array, Array, int],
) -> Array:
    return x + h * f(x, args_f) + g * dW


# ---------------------------------------------------------------------------
# Exact, scalable sampling without replacement
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("n_experiments", "s", "replace"))
def _sample_minibatch_means(
    key: Array,
    Y: Array,
    s: int,
    n_experiments: int,
    denom: float,
    replace: bool,
) -> Array:
    """Return one mini-batch mean for every Monte Carlo path.

    For ``replace=False`` we use Floyd's algorithm.  Its work depends on s,
    not on m, so exact without-replacement sampling remains practical when m
    is large and s is small.  The returned quantity already includes the
    posterior-mean scaling factor ``1/denom``.
    """
    m = Y.shape[0]

    if replace:
        indices = jax.random.randint(key, (n_experiments, s), 0, m)
        return jnp.mean(Y[indices], axis=1) / denom

    if s == m:
        return jnp.full((n_experiments,), jnp.mean(Y) / denom)

    def floyd_indices(k: int) -> Array:
        # Floyd's algorithm generates a uniform subset of size k from
        # {0,...,m-1}.  We sample the smaller of the batch and its complement
        # so near-full batches remain efficient as well.
        positions = jnp.arange(k)
        selected = jnp.full((n_experiments, k), -1, dtype=jnp.int32)

        def body(i: int, current: Array) -> Array:
            draw_key = jax.random.fold_in(key, i)
            upper = m - k + i  # draw from {0,...,upper}
            candidate = jax.random.randint(
                draw_key, (n_experiments,), 0, upper + 1
            )
            already_used = jnp.any(
                (current == candidate[:, None])
                & (positions[None, :] < i),
                axis=1,
            )
            chosen = jnp.where(already_used, upper, candidate)
            return current.at[:, i].set(chosen)

        return jax.lax.fori_loop(0, k, body, selected)

    if s <= m // 2:
        indices = floyd_indices(s)
        batch_mean = jnp.mean(Y[indices], axis=1)
    else:
        complement_size = m - s
        excluded = floyd_indices(complement_size)
        batch_sum = jnp.sum(Y) - jnp.sum(Y[excluded], axis=1)
        batch_mean = batch_sum / s

    return batch_mean / denom


# ---------------------------------------------------------------------------
# Coupled h / h/2 simulations used by the step-doubling test
# ---------------------------------------------------------------------------
@partial(
    jax.jit,
    static_argnames=("n_experiments", "n_coarse", "f", "func"),
)
def _coupled_euler_values(
    key: Array,
    n_experiments: int,
    n_coarse: int,
    h: float,
    f: Drift,
    g: float,
    initial: float,
    mu_post: Array,
    var_post: Array,
    m: int,
    func: Observable,
) -> tuple[Array, Array]:
    h_half = h / 2.0
    dW_half = jnp.sqrt(h_half) * jax.random.normal(
        key, (n_coarse, 2, n_experiments)
    )
    dW_coarse = jnp.sum(dW_half, axis=1)
    args_f = (mu_post, var_post, m)
    x0 = jnp.full((n_experiments,), initial)

    def coarse_step(x: Array, dW_t: Array):
        return euler_onestep(x, h, dW_t, f, g, args_f), None

    def fine_step(x: Array, dW_t: Array):
        return euler_onestep(x, h_half, dW_t, f, g, args_f), None

    x_h, _ = jax.lax.scan(coarse_step, x0, dW_coarse)
    x_half, _ = jax.lax.scan(
        fine_step, x0, dW_half.reshape(2 * n_coarse, n_experiments)
    )
    return func(x_h), func(x_half)


@partial(
    jax.jit,
    static_argnames=(
        "n_experiments",
        "n_coarse",
        "s",
        "replace",
        "f",
        "func",
    ),
)
def _coupled_sgld_values(
    key: Array,
    n_experiments: int,
    n_coarse: int,
    h: float,
    f: Drift,
    g: float,
    initial: float,
    s: int,
    sig_th: float,
    sig_y: float,
    Y: Array,
    var_post: Array,
    func: Observable,
    replace: bool,
) -> tuple[Array, Array]:
    h_half = h / 2.0
    m = Y.shape[0]
    denom = (sig_y**2 / sig_th**2) + 1.0
    x0 = jnp.full((n_experiments,), initial)

    def block_step(carry, _):
        x_h, x_half, loop_key = carry
        loop_key, brownian_key, mb1_key, mb2_key = jax.random.split(loop_key, 4)

        dW_half = jnp.sqrt(h_half) * jax.random.normal(
            brownian_key, (2, n_experiments)
        )
        dW_h = jnp.sum(dW_half, axis=0)

        mu_1 = _sample_minibatch_means(
            mb1_key, Y, s, n_experiments, denom, replace
        )
        mu_2 = _sample_minibatch_means(
            mb2_key, Y, s, n_experiments, denom, replace
        )

        # The coarse level shares the first fine-step mini-batch.  Each level
        # retains the correct one-step marginal, while the coupling reduces
        # noise in the level-difference diagnostic.
        x_h_next = euler_onestep(
            x_h, h, dW_h, f, g, (mu_1, var_post, m)
        )
        x_half_1 = euler_onestep(
            x_half, h_half, dW_half[0], f, g, (mu_1, var_post, m)
        )
        x_half_2 = euler_onestep(
            x_half_1, h_half, dW_half[1], f, g, (mu_2, var_post, m)
        )
        return (x_h_next, x_half_2, loop_key), None

    (x_h, x_half, _), _ = jax.lax.scan(
        block_step,
        (x0, x0, key),
        xs=None,
        length=n_coarse,
    )
    return func(x_h), func(x_half)


def _level_statistics(values_h: Array, values_half: Array) -> tuple[float, float]:
    level_mse = float(jax.device_get(jnp.mean((values_h - values_half) ** 2)))
    variance_h = float(jax.device_get(jnp.var(values_h, ddof=1)))
    return level_mse, variance_h


def find_stepsize_em(
    key: Array,
    epsilon: float,
    n_experiments: int,
    T: float,
    f: Drift,
    g: float,
    initial: float,
    mu_post: Array,
    var_post: Array,
    m: int,
    func: Observable,
    max_refinements: int = 24,
) -> tuple[float, float, float]:
    """Empirical step-doubling rule from the manuscript for EM.

    Start with h=4*epsilon.  Compare coupled h and h/2 paths.  If the mean
    squared level difference exceeds epsilon^2/2, halve h and repeat.  The
    *coarse* accepted h is returned, matching Algorithm 1 in the manuscript.
    """
    threshold_sq = epsilon**2 / 2.0
    h = 4.0 * epsilon

    for level in range(max_refinements):
        n_coarse = max(1, math.ceil(T / h))
        if 2 * n_coarse > _MAX_FINE_STEPS:
            raise RuntimeError(
                "EM step-size search exceeded the safety limit of "
                f"{_MAX_FINE_STEPS:,} fine steps (h={h:.3e}, T={T:.3e})."
            )
        pair_key = jax.random.fold_in(key, level)
        values_h, values_half = _coupled_euler_values(
            pair_key,
            n_experiments,
            n_coarse,
            h,
            f,
            g,
            initial,
            mu_post,
            var_post,
            m,
            func,
        )
        level_mse, variance_h = _level_statistics(values_h, values_half)
        if math.isfinite(level_mse) and math.isfinite(variance_h):
            if level_mse <= threshold_sq:
                return h, max(variance_h, 0.0), level_mse
        h /= 2.0

    raise RuntimeError("EM step-size search did not converge.")


def find_stepsize_sgld(
    key: Array,
    epsilon: float,
    n_experiments: int,
    T: float,
    f: Drift,
    g: float,
    initial: float,
    s: int,
    sig_th: float,
    sig_y: float,
    Y: Array,
    var_post: Array,
    func: Observable,
    replace: bool,
    max_refinements: int = 24,
) -> tuple[float, float, float]:
    """Empirical step-doubling rule from the manuscript for SGLD."""
    threshold_sq = epsilon**2 / 2.0
    h = 4.0 * epsilon

    for level in range(max_refinements):
        n_coarse = max(1, math.ceil(T / h))
        if 2 * n_coarse > _MAX_FINE_STEPS:
            raise RuntimeError(
                "SGLD step-size search exceeded the safety limit of "
                f"{_MAX_FINE_STEPS:,} fine steps (h={h:.3e}, T={T:.3e})."
            )
        pair_key = jax.random.fold_in(key, level)
        values_h, values_half = _coupled_sgld_values(
            pair_key,
            n_experiments,
            n_coarse,
            h,
            f,
            g,
            initial,
            s,
            sig_th,
            sig_y,
            Y,
            var_post,
            func,
            replace,
        )
        level_mse, variance_h = _level_statistics(values_h, values_half)
        if math.isfinite(level_mse) and math.isfinite(variance_h):
            if level_mse <= threshold_sq:
                return h, max(variance_h, 0.0), level_mse
        h /= 2.0

    raise RuntimeError("SGLD step-size search did not converge.")


# ---------------------------------------------------------------------------
# Terminal estimators at a fixed h
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("n_paths", "n_steps", "f", "func"))
def _euler_terminal_values(
    key: Array,
    n_paths: int,
    n_steps: int,
    h: float,
    f: Drift,
    g: float,
    initial: float,
    mu_post: Array,
    var_post: Array,
    m: int,
    func: Observable,
) -> Array:
    dW = jnp.sqrt(h) * jax.random.normal(key, (n_steps, n_paths))
    x0 = jnp.full((n_paths,), initial)
    args_f = (mu_post, var_post, m)

    def step(x, dW_t):
        return euler_onestep(x, h, dW_t, f, g, args_f), None

    xT, _ = jax.lax.scan(step, x0, dW)
    return func(xT)


@partial(
    jax.jit,
    static_argnames=("n_paths", "n_steps", "s", "replace", "f", "func"),
)
def _sgld_terminal_values(
    key: Array,
    n_paths: int,
    n_steps: int,
    h: float,
    f: Drift,
    g: float,
    initial: float,
    s: int,
    sig_th: float,
    sig_y: float,
    Y: Array,
    var_post: Array,
    func: Observable,
    replace: bool,
) -> Array:
    m = Y.shape[0]
    denom = (sig_y**2 / sig_th**2) + 1.0
    x0 = jnp.full((n_paths,), initial)

    def step(carry, _):
        x, loop_key = carry
        loop_key, brownian_key, mb_key = jax.random.split(loop_key, 3)
        dW = jnp.sqrt(h) * jax.random.normal(brownian_key, (n_paths,))
        mu_batch = _sample_minibatch_means(
            mb_key, Y, s, n_paths, denom, replace
        )
        x_next = euler_onestep(
            x, h, dW, f, g, (mu_batch, var_post, m)
        )
        return (x_next, loop_key), None

    (xT, _), _ = jax.lax.scan(
        step, (x0, key), xs=None, length=n_steps
    )
    return func(xT)


def _batched_mean(
    simulator: Callable,
    key: Array,
    n_paths: int,
    *args,
) -> float:
    """Return the mean of ``simulator`` outputs over ``n_paths`` in chunks."""
    remaining = int(n_paths)
    total = 0
    sum_values = 0.0
    while remaining > 0:
        batch = min(remaining, _MAX_BATCH)
        key, subkey = jax.random.split(key)
        values = simulator(subkey, batch, *args)
        sum_values += float(jax.device_get(jnp.sum(values)))
        total += batch
        remaining -= batch
    return sum_values / total


# ---------------------------------------------------------------------------
# Public cost experiment
# ---------------------------------------------------------------------------
def costs(
    key: Array,
    m: int,
    s: int,
    epsilon: float,
    sig_th: float,
    sig_y: float,
    func: Observable,
    f: Drift = f_ou,
    g: float | None = None,
    initial: float = 1.0,
    n_experiments: int = 100,
    n: int = 5,
    replace: bool = False,
    horizon_factor: float = 3.0,
    max_validation_paths: int = 4096,
) -> tuple[float, float, float, float]:
    """Estimate empirical RMSE and the paper's leading operation-count cost.

    Parameters
    ----------
    m, s, epsilon
        Data size, mini-batch size, and target RMSE.  The paper's default
        SGLD convention is 1 <= s <= m with sampling without replacement.
    n_experiments
        Pilot path count used by the step-doubling criterion and variance
        estimate.
    n
        Number of indepently regenerated datasets/outer repetitions.
    replace
        Whether mini-batch sampling is with replacement.  Default ``False``
        matches the manuscript.
    horizon_factor
        The numerical protocol uses T = ceil(horizon_factor*log(1/epsilon));
        the manuscript uses 3.
    max_validation_paths
        The theoretical Monte Carlo path count N can be very large.  The
        reported operation-count cost always uses the full N, while the RMSE
        diagnostic is estimated with at most this many paths to keep a run
        practical.  Set to 0 to simulate all N paths.
    """
    if not (0.0 < float(epsilon) < 1.0):
        raise ValueError("epsilon must lie in (0, 1).")
    if int(m) < 2:
        raise ValueError("m must be at least 2.")
    if not (1 <= int(s) <= int(m)):
        raise ValueError("s must satisfy 1 <= s <= m.")
    if n_experiments < 2:
        raise ValueError("n_experiments must be at least 2.")
    if n < 1:
        raise ValueError("n must be at least 1.")
    if sig_th <= 0 or sig_y <= 0:
        raise ValueError("sig_th and sig_y must be positive.")
    if horizon_factor <= 0:
        raise ValueError("horizon_factor must be positive.")

    if g is None:
        g = 1.0 / math.sqrt(m)

    T = float(math.ceil(horizon_factor * math.log(1.0 / epsilon)))

    euler_sq_errors: list[float] = []
    sgld_sq_errors: list[float] = []
    euler_costs: list[float] = []
    sgld_costs: list[float] = []

    for rep in range(n):
        rep_key = jax.random.fold_in(key, rep)
        (
            theta_key,
            noise_key,
            em_step_key,
            sgld_step_key,
            em_final_key,
            sgld_final_key,
        ) = jax.random.split(rep_key, 6)

        # Scaled prior from the manuscript.
        theta = (sig_th / math.sqrt(m)) * jax.random.normal(theta_key, ())
        Y = theta + sig_y * jax.random.normal(noise_key, (m,))

        var_post, mu_post = var_mu_comp(Y, sig_th, sig_y)
        target = float(jax.device_get(ref_im_sol(mu_post, var_post, func)))

        h_em, var_em, _ = find_stepsize_em(
            em_step_key,
            epsilon,
            n_experiments,
            T,
            f,
            g,
            initial,
            mu_post,
            var_post,
            m,
            func,
        )
        if (not replace) and s == m:
            # With a full batch and without-replacement sampling, SGLD is
            # exactly the Euler scheme.  Reuse the EM calibration so the
            # numerical implementation respects this identity rather than
            # introducing artificial differences from pilot Monte Carlo noise.
            h_sgld, var_sgld = h_em, var_em
        else:
            h_sgld, var_sgld, _ = find_stepsize_sgld(
                sgld_step_key,
                epsilon,
                n_experiments,
                T,
                f,
                g,
                initial,
                s,
                sig_th,
                sig_y,
                Y,
                var_post,
                func,
                replace,
            )

        n_paths_em = max(1, math.ceil(2.0 * var_em / epsilon**2))
        n_paths_sgld = max(1, math.ceil(2.0 * var_sgld / epsilon**2))
        n_steps_em = math.ceil(T / h_em)
        n_steps_sgld = math.ceil(T / h_sgld)

        if max_validation_paths and max_validation_paths > 0:
            n_run_em = min(n_paths_em, max_validation_paths)
            n_run_sgld = min(n_paths_sgld, max_validation_paths)
        else:
            n_run_em = n_paths_em
            n_run_sgld = n_paths_sgld

        em_mean = _batched_mean(
            _euler_terminal_values,
            em_final_key,
            n_run_em,
            n_steps_em,
            h_em,
            f,
            g,
            initial,
            mu_post,
            var_post,
            m,
            func,
        )
        if (not replace) and s == m:
            sgld_mean = em_mean
        else:
            sgld_mean = _batched_mean(
                _sgld_terminal_values,
                sgld_final_key,
                n_run_sgld,
                n_steps_sgld,
                h_sgld,
                f,
                g,
                initial,
                s,
                sig_th,
                sig_y,
                Y,
                var_post,
                func,
                replace,
            )

        euler_sq_errors.append((em_mean - target) ** 2)
        sgld_sq_errors.append((sgld_mean - target) ** 2)

        # The cost model in the manuscript counts drift evaluations, not wall
        # time: m data contributions per EM step and s per SGLD step.
        euler_costs.append(float(n_paths_em) * n_steps_em * m)
        sgld_costs.append(float(n_paths_sgld) * n_steps_sgld * s)

    euler_rmse = math.sqrt(math.fsum(euler_sq_errors) / n)
    sgld_rmse = math.sqrt(math.fsum(sgld_sq_errors) / n)
    mean_euler_cost = math.fsum(euler_costs) / n
    mean_sgld_cost = math.fsum(sgld_costs) / n

    return euler_rmse, mean_euler_cost, sgld_rmse, mean_sgld_cost
