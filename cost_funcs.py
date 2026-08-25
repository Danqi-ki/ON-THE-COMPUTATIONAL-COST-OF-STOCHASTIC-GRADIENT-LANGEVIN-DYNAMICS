# -*- coding: utf-8 -*-


import math
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from functools import partial

# Reusable thread pool for overlapping Euler/SGLD work.
_THREAD_POOL = ThreadPoolExecutor(max_workers=2)

# Maximum number of experiments to process in one JIT call.
# Keeps peak memory under ~2 GB regardless of epsilon.
_MAX_BATCH = 4096

# Maximum number of time steps before we stop refining the stepsize.
# Prevents the stepsize search from producing absurdly small h values
# that blow up memory (n1 = T/h grows without bound).
_MAX_N1 = 1_000_000


# ------------------------------
# Posterior mean/variance for mu
# ------------------------------
def var_mu_comp(Y, sig_th, sig_y):
    m = Y.shape[0]
    term_in_paren = (sig_y**2 / sig_th**2) + 1.0
    sum_Y = jnp.sum(Y)
    mu_post = (sum_Y / m) / term_in_paren

    precision_sum = 1.0 / (sig_th**2) + 1.0 / (sig_y**2)
    var_post = (1.0 / m) / precision_sum
    return var_post, mu_post


# ------------------------------
# Reference invariant-measure integral (numerical)
# ------------------------------
def ref_im_sol(mu_post, var_post, func):
    x = jnp.arange(-2.0, 2.0, 0.0001) + mu_post
    y = func(x) * jnp.exp(-0.5 * (x - mu_post) ** 2 / var_post) / jnp.sqrt(
        2.0 * jnp.pi * var_post
    )
    return trapezoid(y, x)


# ------------------------------
# Drift for OU-like process
# ------------------------------
def f_ou(x, args):
    """
    Ornstein–Uhlenbeck drift used in your code:
      f(x) = -(x - mu) / (2*sigma2) / m
    args = (mu, sigma2, m)
    """
    mu, sigma2, m = args
    return -(x - mu) / (2.0 * sigma2) / m


# ------------------------------
# One Euler step
# ------------------------------
def euler_onestep(x, h, dW, f, g_c, args_f):
    return x + h * f(x, args_f) + g_c * dW


# ------------------------------
# Minibatch mu(t, experiment) generator
# ------------------------------
@partial(jax.jit, static_argnames=("num_experiments", "num_iterations", "replace", "s"))
def sampling_mu_sgd_n_fast(
    key,
    Y,
    s,
    sig_th,
    sig_y,
    num_experiments: int,
    num_iterations: int,
    replace: bool = True,
):
    """
    Returns mu_sgd with shape (num_iterations, num_experiments).

    replace=True (recommended): scalable for m up to 1e6+ if s is moderate.
    replace=False: may be very expensive for large m.
    """
    m = Y.shape[0]
    denom = (sig_y**2 / sig_th**2) + 1.0

    # One key per (t, e)
    keys = jax.random.split(key, num_iterations * num_experiments)
    keys = keys.reshape(num_iterations, num_experiments, 2)

    def sample_one(k):
        if replace:
            idx = jax.random.randint(k, (s,), 0, m)
        else:
            # WARNING: expensive when m is large
            idx = jax.random.choice(k, m, (s,), replace=False)
        return jnp.mean(jnp.take(Y, idx)) / denom

    mu = jax.vmap(lambda row: jax.vmap(sample_one)(row))(keys)
    return mu


# ------------------------------
# Euler functional generator (constant mu_post)
# ------------------------------
@partial(jax.jit, static_argnames=("n_experiments", "n1", "f", "func"))
def functional_euler_generator_fast(
    key,
    n_experiments: int,
    n1: int,
    h,
    f,
    g,
    initial,
    mu_post,
    var_post,
    m,
    func,
):
    # Brownian increments: (n1, n_experiments)
    dW = jnp.sqrt(h) * jax.random.normal(key, (n1, n_experiments))

    args_f = (mu_post, var_post, m)
    x0 = jnp.full((n_experiments,), initial)

    def step(x, dW_t):
        x_next = euler_onestep(x, h, dW_t, f, g, args_f)
        return x_next, None

    xT, _ = jax.lax.scan(step, x0, dW)
    vals = func(xT)
    return jnp.mean(vals), jnp.var(vals)


# ------------------------------
# SGD functional generator (time-varying mu_sgd[t, :])
# ------------------------------
@partial(jax.jit, static_argnames=("n_experiments", "n1", "replace", "f", "func", "s"))
def functional_sgd_generator_fast(
    key,
    n_experiments: int,
    n1: int,
    h,
    f,
    g,
    initial,
    s,
    sig_th,
    sig_y,
    Y,
    mu_post,   # not used in drift here, kept for signature similarity
    var_post,
    func,
    replace: bool = True,
):
    m = Y.shape[0]
    denom = (sig_y**2 / sig_th**2) + 1.0
    sqrt_h = jnp.sqrt(h)

    # Pre-split all keys for BM and mu sampling (just key material, not data)
    keys = jax.random.split(key, 2 * n1)
    bm_keys = keys[:n1]
    mu_keys = keys[n1:]

    x0 = jnp.full((n_experiments,), initial)

    def _sample_mu_one_step(mu_key, n_exp):
        """Sample minibatch mu for one timestep, all experiments."""
        sub_keys = jax.random.split(mu_key, n_exp)
        def sample_one(k):
            if replace:
                idx = jax.random.randint(k, (s,), 0, m)
            else:
                # Exact sampling without replacement. This can be expensive
                # when m is very large, so replace=True remains recommended.
                idx = jax.random.choice(k, m, shape=(s,), replace=False)
            return jnp.mean(jnp.take(Y, idx)) / denom
        return jax.vmap(sample_one)(sub_keys)

    def step(x, inp):
        bm_key, mu_key = inp
        dW_t = sqrt_h * jax.random.normal(bm_key, (n_experiments,))
        mu_t = _sample_mu_one_step(mu_key, n_experiments)
        args_f = (mu_t, var_post, m)
        x_next = euler_onestep(x, h, dW_t, f, g, args_f)
        return x_next, None

    xT, _ = jax.lax.scan(step, x0, (bm_keys, mu_keys))
    vals = func(xT)
    return jnp.mean(vals), jnp.var(vals)


# ------------------------------
# Batched wrappers (avoid OOM for large N)
# ------------------------------
def functional_euler_batched(key, N, n1, h, f, g, initial, mu_post, var_post, m, func):
    """Run Euler in chunks of _MAX_BATCH, return (mean, var) over all N paths."""
    sum_vals = 0.0
    sum_sq = 0.0
    total = 0
    remaining = N
    while remaining > 0:
        batch = min(remaining, _MAX_BATCH)
        key, subkey = jax.random.split(key)
        mean_b, var_b = functional_euler_generator_fast(
            subkey, batch, n1, h, f, g, initial, mu_post, var_post, m, func
        )
        # Welford-style accumulation: track sum and sum-of-squares
        sum_vals += float(mean_b) * batch
        sum_sq += (float(var_b) + float(mean_b)**2) * batch
        total += batch
        remaining -= batch
    overall_mean = sum_vals / total
    # Round-off can produce a tiny negative value for a theoretically
    # non-negative variance.
    overall_var = max(0.0, sum_sq / total - overall_mean**2)
    return float(overall_mean), float(overall_var)


def functional_sgd_batched(key, N, n1, h, f, g, initial, s, sig_th, sig_y, Y,
                           mu_post, var_post, func, replace=True):
    """Run SGD in chunks of _MAX_BATCH, return (mean, var) over all N paths."""
    sum_vals = 0.0
    sum_sq = 0.0
    total = 0
    remaining = N
    while remaining > 0:
        batch = min(remaining, _MAX_BATCH)
        key, subkey = jax.random.split(key)
        mean_b, var_b = functional_sgd_generator_fast(
            subkey, batch, n1, h, f, g, initial, s, sig_th, sig_y, Y,
            mu_post, var_post, func, replace
        )
        sum_vals += float(mean_b) * batch
        sum_sq += (float(var_b) + float(mean_b)**2) * batch
        total += batch
        remaining -= batch
    overall_mean = sum_vals / total
    # Round-off can produce a tiny negative value for a theoretically
    # non-negative variance.
    overall_var = max(0.0, sum_sq / total - overall_mean**2)
    return float(overall_mean), float(overall_var)


# ------------------------------
# Coupled coarse/fine functional generators
# ------------------------------
def _validate_refinement_pair(h_coarse, h_fine):
    """Return the integer refinement factor for a nested stepsize pair."""
    ratio = float(h_coarse) / float(h_fine)
    refinement = int(round(ratio))
    if refinement < 2 or not math.isclose(
        ratio, refinement, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(
            "Coupled stepsize search requires h_coarse/h_fine to be an "
            f"integer >= 2; got h_coarse={h_coarse}, h_fine={h_fine}."
        )
    return refinement


def coupled_functional_euler_generator(
    n_experiments,
    T,
    h_coarse,
    h_fine,
    f,
    g,
    initial,
    mu_post,
    var_post,
    m,
    func,
    key,
):
    """Return pathwise coupled Euler functional values at coarse/fine levels."""
    refinement = _validate_refinement_pair(h_coarse, h_fine)
    n_coarse = max(1, math.ceil(float(T) / float(h_coarse)))
    n_fine = n_coarse * refinement

    if n_fine > _MAX_N1:
        raise RuntimeError(
            f"Fine Euler path needs {n_fine} steps, exceeding _MAX_N1={_MAX_N1}."
        )

    dW_fine = jnp.sqrt(h_fine) * jax.random.normal(
        key, (n_coarse, refinement, n_experiments)
    )
    dW_coarse = jnp.sum(dW_fine, axis=1)

    args_f = (mu_post, var_post, m)
    x0 = jnp.full((n_experiments,), initial)

    def coarse_step(x, dW_t):
        x_next = euler_onestep(x, h_coarse, dW_t, f, g, args_f)
        return x_next, None

    def fine_step(x, dW_t):
        x_next = euler_onestep(x, h_fine, dW_t, f, g, args_f)
        return x_next, None

    x_coarse, _ = jax.lax.scan(coarse_step, x0, dW_coarse)
    x_fine, _ = jax.lax.scan(
        fine_step,
        x0,
        dW_fine.reshape(n_fine, n_experiments),
    )

    return func(x_coarse), func(x_fine)


def coupled_functional_sgd_generator_n(
    n_experiments,
    T,
    h_coarse,
    h_fine,
    f,
    g,
    initial,
    s,
    sig_th,
    sig_y,
    Y,
    mu_post,
    var_post,
    func,
    key,
    replace=True,
):
    """
    Return pathwise coupled SGLD functional values at coarse/fine levels.

    Brownian increments are nested.  The coarse step uses the first
    minibatch estimate from the corresponding fine-step block; this gives
    both levels the correct one-step minibatch marginal while sharing as
    much randomness as possible.
    """
    del mu_post  # Kept only for interface compatibility.

    refinement = _validate_refinement_pair(h_coarse, h_fine)
    n_coarse = max(1, math.ceil(float(T) / float(h_coarse)))
    n_fine = n_coarse * refinement

    if n_fine > _MAX_N1:
        raise RuntimeError(
            f"Fine SGLD path needs {n_fine} steps, exceeding _MAX_N1={_MAX_N1}."
        )

    m_data = Y.shape[0]
    if s > m_data and not replace:
        raise ValueError("s cannot exceed m when replace=False.")

    denom = (sig_y**2 / sig_th**2) + 1.0
    x0 = jnp.full((n_experiments,), initial)

    def sample_minibatch_means(sample_key):
        if replace:
            indices = jax.random.randint(
                sample_key,
                (refinement, n_experiments, s),
                0,
                m_data,
            )
        else:
            sample_keys = jax.random.split(
                sample_key, refinement * n_experiments
            ).reshape(refinement, n_experiments, 2)

            def sample_one(k):
                return jax.random.choice(k, m_data, (s,), replace=False)

            indices = jax.vmap(
                lambda row: jax.vmap(sample_one)(row)
            )(sample_keys)

        return jnp.mean(jnp.take(Y, indices), axis=-1) / denom

    def coupled_block_step(carry, _):
        x_coarse, x_fine, loop_key = carry
        loop_key, brownian_key, minibatch_key = jax.random.split(loop_key, 3)

        dW_fine = jnp.sqrt(h_fine) * jax.random.normal(
            brownian_key, (refinement, n_experiments)
        )
        mu_fine = sample_minibatch_means(minibatch_key)

        dW_coarse = jnp.sum(dW_fine, axis=0)
        mu_coarse = mu_fine[0]
        args_coarse = (mu_coarse, var_post, m_data)
        x_coarse_next = euler_onestep(
            x_coarse, h_coarse, dW_coarse, f, g, args_coarse
        )

        def fine_substep(x, inputs):
            dW_t, mu_t = inputs
            args_fine = (mu_t, var_post, m_data)
            x_next = euler_onestep(x, h_fine, dW_t, f, g, args_fine)
            return x_next, None

        x_fine_next, _ = jax.lax.scan(
            fine_substep, x_fine, (dW_fine, mu_fine)
        )

        return (x_coarse_next, x_fine_next, loop_key), None

    (x_coarse, x_fine, _), _ = jax.lax.scan(
        coupled_block_step,
        (x0, x0, key),
        xs=None,
        length=n_coarse,
    )

    return func(x_coarse), func(x_fine)


# ------------------------------
# Step-size search (host-side loop — original sequential version)
# ------------------------------
def _to_float(x):
    return float(jax.device_get(x))

def find_stepsize_EM(
    threshold_sq,
    n_experiments,
    T,
    h_list,
    f,
    g,
    initial,
    mu_post,
    var_post,
    m,
    func,
    key,
):
    """Select the first fine stepsize whose empirical level MSE is acceptable."""
    if n_experiments < 2:
        raise ValueError("n_experiments must be at least 2 to estimate variance.")

    for i in range(len(h_list) - 1):
        h_coarse = h_list[i]
        h_fine = h_list[i + 1]
        pair_key = jax.random.fold_in(key, i)

        values_coarse, values_fine = coupled_functional_euler_generator(
            n_experiments,
            T,
            h_coarse,
            h_fine,
            f,
            g,
            initial,
            mu_post,
            var_post,
            m,
            func,
            pair_key,
        )

        level_mse = jnp.mean(jnp.square(values_coarse - values_fine))
        variance_fine = jnp.var(values_fine, ddof=1)

        valid = bool(
            jax.device_get(
                jnp.isfinite(level_mse) & jnp.isfinite(variance_fine)
            )
        )

        if valid and float(level_mse) <= float(threshold_sq):
            return h_fine, variance_fine, level_mse

    raise RuntimeError(
        "No EM stepsize in h_list satisfies the empirical strong-error criterion."
    )


def find_stepsize_sgd_n(
    threshold_sq,
    n_experiments,
    T,
    h_list,
    f,
    g,
    initial,
    s,
    sig_th,
    sig_y,
    Y,
    mu_post,
    var_post,
    func,
    key,
    replace=True,
):
    """Select the first fine SGLD stepsize whose empirical level MSE is acceptable."""
    if n_experiments < 2:
        raise ValueError("n_experiments must be at least 2 to estimate variance.")

    for i in range(len(h_list) - 1):
        h_coarse = h_list[i]
        h_fine = h_list[i + 1]
        pair_key = jax.random.fold_in(key, i)

        values_coarse, values_fine = coupled_functional_sgd_generator_n(
            n_experiments,
            T,
            h_coarse,
            h_fine,
            f,
            g,
            initial,
            s,
            sig_th,
            sig_y,
            Y,
            mu_post,
            var_post,
            func,
            pair_key,
            replace=replace,
        )

        level_mse = jnp.mean(jnp.square(values_coarse - values_fine))
        variance_fine = jnp.var(values_fine, ddof=1)

        valid = bool(
            jax.device_get(
                jnp.isfinite(level_mse) & jnp.isfinite(variance_fine)
            )
        )

        if valid and float(level_mse) <= float(threshold_sq):
            return h_fine, variance_fine, level_mse

    raise RuntimeError(
        "No SGLD stepsize in h_list satisfies the empirical strong-error criterion."
    )


# ------------------------------
# Vectorized step-size search — runs ALL h candidates in one padded lax.scan
# ------------------------------
@partial(jax.jit, static_argnames=("n_experiments", "max_n1", "f", "func"))
def _euler_all_candidates(key, n_experiments, max_n1, h_arr, n1_arr, f, g,
                          initial, mu_post, var_post, m, func):
    """Run Euler for all h candidates in parallel via vmap over padded scans."""
    n_cand = h_arr.shape[0]
    keys = jax.random.split(key, n_cand)

    def _run_one(k, h, n1_actual):
        dW = jnp.sqrt(h) * jax.random.normal(k, (max_n1, n_experiments))
        args_f = (mu_post, var_post, m)
        x0 = jnp.full((n_experiments,), initial)

        def step(carry, t):
            x, = carry
            # Mask: only apply step if t < n1_actual
            active = (t < n1_actual).astype(dW.dtype)
            dW_t = dW[t]
            x_next = x + active * (h * f(x, args_f) + g * dW_t)
            return (x_next,), None

        (xT,), _ = jax.lax.scan(step, (x0,), jnp.arange(max_n1))
        vals = func(xT)
        return jnp.mean(vals), jnp.var(vals)

    means, variances = jax.vmap(_run_one)(keys, h_arr, n1_arr)
    return means, variances


@partial(jax.jit, static_argnames=("n_experiments", "max_n1", "f", "func", "s"))
def _sgd_all_candidates(key, n_experiments, max_n1, h_arr, n1_arr, f, g,
                        initial, s, sig_th, sig_y, Y, mu_post, var_post, func):
    """Run SGD for all h candidates in parallel via vmap over padded scans."""
    n_cand = h_arr.shape[0]
    m_data = Y.shape[0]
    denom = (sig_y**2 / sig_th**2) + 1.0
    keys = jax.random.split(key, n_cand)

    def _run_one(k, h, n1_actual):
        all_keys = jax.random.split(k, 2 * max_n1)
        bm_keys = all_keys[:max_n1]
        mu_keys = all_keys[max_n1:]
        sqrt_h = jnp.sqrt(h)
        x0 = jnp.full((n_experiments,), initial)

        def _sample_mu(mu_key):
            sub_keys = jax.random.split(mu_key, n_experiments)
            def sample_one(kk):
                idx = jax.random.randint(kk, (s,), 0, m_data)
                return jnp.mean(jnp.take(Y, idx)) / denom
            return jax.vmap(sample_one)(sub_keys)

        def step(carry, t):
            x, = carry
            active = (t < n1_actual).astype(x.dtype)
            dW_t = sqrt_h * jax.random.normal(bm_keys[t], (n_experiments,))
            mu_t = _sample_mu(mu_keys[t])
            args_f = (mu_t, var_post, m_data)
            x_next = x + active * (h * f(x, args_f) + g * dW_t)
            return (x_next,), None

        (xT,), _ = jax.lax.scan(step, (x0,), jnp.arange(max_n1))
        vals = func(xT)
        return jnp.mean(vals), jnp.var(vals)

    means, variances = jax.vmap(_run_one)(keys, h_arr, n1_arr)
    return means, variances


def _pick_converged(means, variances, h_list, threshold):
    """From arrays of means/variances for each h candidate, find first converged pair."""
    n = len(means)
    for i in range(n - 1):
        x1, v1 = float(means[i]), float(variances[i])
        x2, v2 = float(means[i + 1]), float(variances[i + 1])
        if math.isnan(v1) or math.isnan(v2):
            continue
        if abs(x1 - x2) <= threshold:
            return h_list[i], jnp.float32(v2)
    # fallback: last candidate
    return h_list[n - 1], jnp.float32(float(variances[n - 1]))


def find_stepsize_EM_vectorized(
    key, threshold, n_experiments, T, h_list, f, g, initial,
    mu_post, var_post, m, func,
):
    """Vectorized stepsize search for Euler: one JIT call for all candidates."""
    n1_list = [int(T / h) for h in h_list]
    max_n1 = max(n1_list)

    h_arr = jnp.array(h_list, dtype=jnp.float32)
    n1_arr = jnp.array(n1_list, dtype=jnp.int32)

    means, variances = _euler_all_candidates(
        key, n_experiments, max_n1, h_arr, n1_arr, f, g,
        initial, mu_post, var_post, m, func,
    )

    h, v2 = _pick_converged(means, variances, h_list, threshold)
    return h, v2


def find_stepsize_sgd_n_vectorized(
    key, threshold, n_experiments, T, h_list, f, g, initial,
    s, sig_th, sig_y, Y, mu_post, var_post, func,
):
    """Vectorized stepsize search for SGD: one JIT call for all candidates."""
    n1_list = [int(T / h) for h in h_list]
    max_n1 = max(n1_list)

    h_arr = jnp.array(h_list, dtype=jnp.float32)
    n1_arr = jnp.array(n1_list, dtype=jnp.int32)

    means, variances = _sgd_all_candidates(
        key, n_experiments, max_n1, h_arr, n1_arr, f, g,
        initial, s, sig_th, sig_y, Y, mu_post, var_post, func,
    )

    h, v2 = _pick_converged(means, variances, h_list, threshold)
    # SGD uses h/2
    return h / 2.0, v2


# ------------------------------
# Main cost function (n outer experiments)
# ------------------------------
def costs(
    key,
    m: int,
    s: int,
    epsilon: float,
    sig_th: float,
    sig_y: float,
    func,
    f=f_ou,
    g: float = 1.0,
    initial: float = 0.0,
    n_experiments: int = 10**2,
    n: int = 10,
    K: float = 0.5,
    replace: bool = True,
    parallel: bool = True,
    vectorized: bool = False,
):
    """
    Estimate Euler/SGLD RMSE and leading computational cost.

    The public signature deliberately keeps ``key`` as the first argument so
    existing callers such as ``run_fast_fixed_s.py`` remain compatible.
    The new coupled MSE stepsize search returns
    ``(h_fine, variance_fine, level_mse)``.
    """
    if vectorized:
        raise NotImplementedError(
            "vectorized=True still uses the old independent-mean criterion. "
            "Use vectorized=False with the new coupled level-MSE search."
        )
    if not (0.0 < float(epsilon) < 1.0):
        raise ValueError("epsilon must lie in (0, 1).")
    if m < 1 or s < 1:
        raise ValueError("m and s must be positive integers.")
    if not replace and s > m:
        raise ValueError("s cannot exceed m when replace=False.")
    if n_experiments < 2:
        raise ValueError("n_experiments must be at least 2.")

    T = float((1.0 / K) * jnp.log(1.0 / epsilon))

    # Descending dyadic candidates: 2*epsilon, epsilon, epsilon/2, ...
    # Keep only pairs whose fine path does not exceed _MAX_N1.
    raw_h_list = [4.0 * epsilon / (2**i) for i in range(1, 20)]
    h_list = [
        h for h in raw_h_list
        if math.ceil(T / float(h)) <= _MAX_N1
    ]
    if len(h_list) < 2:
        raise RuntimeError(
            "Fewer than two candidate stepsizes satisfy _MAX_N1. "
            "Increase _MAX_N1 or use a larger epsilon."
        )

    threshold_sq = epsilon**2 / 2.0

    euler_error_list = []
    sgd_error_list = []
    euler_cost_list = []
    sgd_cost_list = []

    for j in range(n):

        experiment_key = jax.random.fold_in(key, j)
        (
            data_key,
            em_step_key,
            sgld_step_key,
            em_final_key,
            sgld_final_key,
        ) = jax.random.split(experiment_key, 5)

        theta_key, data_noise_key = jax.random.split(data_key)
        theta = sig_th * jax.random.normal(theta_key, (1,))
        Y = theta + sig_y * jax.random.normal(data_noise_key, (m,))

        var_post, mu_post = var_mu_comp(Y, sig_th, sig_y)
        ans = ref_im_sol(mu_post, var_post, func)

        em_kwargs = dict(
            threshold_sq=threshold_sq,
            n_experiments=n_experiments,
            T=T,
            h_list=h_list,
            f=f,
            g=g,
            initial=initial,
            mu_post=mu_post,
            var_post=var_post,
            m=m,
            func=func,
            key=em_step_key,
        )
        sgld_kwargs = dict(
            threshold_sq=threshold_sq,
            n_experiments=n_experiments,
            T=T,
            h_list=h_list,
            f=f,
            g=g,
            initial=initial,
            s=s,
            sig_th=sig_th,
            sig_y=sig_y,
            Y=Y,
            mu_post=mu_post,
            var_post=var_post,
            func=func,
            key=sgld_step_key,
            replace=replace,
        )

        if parallel:
            fut_euler = _THREAD_POOL.submit(find_stepsize_EM, **em_kwargs)
            fut_sgld = _THREAD_POOL.submit(find_stepsize_sgd_n, **sgld_kwargs)
            h_euler, v_euler, em_level_mse = fut_euler.result()
            h_sgd, v_sgd, sgld_level_mse = fut_sgld.result()
        else:
            h_euler, v_euler, em_level_mse = find_stepsize_EM(**em_kwargs)
            h_sgd, v_sgd, sgld_level_mse = find_stepsize_sgd_n(**sgld_kwargs)

        N_full = max(1, math.ceil(2.0 * float(v_euler) / epsilon**2))
        N_sgd = max(1, math.ceil(2.0 * float(v_sgd) / epsilon**2))

        n_step_euler = math.ceil(T / float(h_euler))
        n_step_sgd = math.ceil(T / float(h_sgd))

        # Preserve the fast-runner safeguard: estimate RMSE with at most one
        # _MAX_BATCH-sized simulation, while reporting theoretical N-based cost.
        N_run_euler = min(N_full, _MAX_BATCH)
        N_run_sgd = min(N_sgd, _MAX_BATCH)

        if parallel:
            fut_euler_sim = _THREAD_POOL.submit(
                functional_euler_batched,
                em_final_key,
                N_run_euler,
                n_step_euler,
                h_euler,
                f,
                g,
                initial,
                mu_post,
                var_post,
                m,
                func,
            )
            fut_sgld_sim = _THREAD_POOL.submit(
                functional_sgd_batched,
                sgld_final_key,
                N_run_sgd,
                n_step_sgd,
                h_sgd,
                f,
                g,
                initial,
                s,
                sig_th,
                sig_y,
                Y,
                mu_post,
                var_post,
                func,
                replace,
            )
            full_euler_sol, _ = fut_euler_sim.result()
            full_sgd_sol, _ = fut_sgld_sim.result()
        else:
            full_euler_sol, _ = functional_euler_batched(
                em_final_key,
                N_run_euler,
                n_step_euler,
                h_euler,
                f,
                g,
                initial,
                mu_post,
                var_post,
                m,
                func,
            )
            full_sgd_sol, _ = functional_sgd_batched(
                sgld_final_key,
                N_run_sgd,
                n_step_sgd,
                h_sgd,
                f,
                g,
                initial,
                s,
                sig_th,
                sig_y,
                Y,
                mu_post,
                var_post,
                func,
                replace,
            )

        euler_error_list.append((full_euler_sol - ans) ** 2)
        sgd_error_list.append((full_sgd_sol - ans) ** 2)

        # Store costs as floating-point values immediately. For large m,
        # N_full * n_step_euler * m can exceed the fixed-width integer range
        # used by NumPy/JAX array conversion on Windows, even though Python's
        # own integer type can represent the value.
        euler_cost = float(N_full) * float(n_step_euler) * float(m)
        sgd_cost = float(N_sgd) * float(n_step_sgd) * float(s)

        if not math.isfinite(euler_cost):
            raise FloatingPointError(
                "Non-finite Euler cost detected: "
                f"m={m}, s={s}, epsilon={epsilon}, "
                f"N_full={N_full}, n_step_euler={n_step_euler}"
            )
        if not math.isfinite(sgd_cost):
            raise FloatingPointError(
                "Non-finite SGLD cost detected: "
                f"m={m}, s={s}, epsilon={epsilon}, "
                f"N_sgd={N_sgd}, n_step_sgd={n_step_sgd}"
            )

        euler_cost_list.append(euler_cost)
        sgd_cost_list.append(sgd_cost)

    # Error terms are JAX scalar arrays. Stack them explicitly rather than
    # relying on generic Python-list conversion.
    euler_rmse = jnp.sqrt(jnp.mean(jnp.stack(euler_error_list)))
    sgd_rmse = jnp.sqrt(jnp.mean(jnp.stack(sgd_error_list)))

    # Costs are ordinary Python floats. math.fsum avoids converting large
    # Python integers into fixed-width JAX/NumPy integer arrays and provides
    # a numerically stable sum.
    mean_euler_cost = math.fsum(euler_cost_list) / len(euler_cost_list)
    mean_sgd_cost = math.fsum(sgd_cost_list) / len(sgd_cost_list)

    return (
        euler_rmse,
        mean_euler_cost,
        sgd_rmse,
        mean_sgd_cost,
    )

