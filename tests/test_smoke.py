import math

import jax
import jax.numpy as jnp

from src.cost_funcs import (
    _sample_minibatch_means,
    costs,
    observable_sin,
    ref_im_sol,
    var_mu_comp,
)


def test_scaled_prior_posterior_formula():
    y = jnp.array([1.0, 2.0, 3.0, 4.0])
    var_post, mu_post = var_mu_comp(y, sig_th=2.0, sig_y=1.0)
    expected_mu = float(jnp.mean(y)) / (1.0 / 4.0 + 1.0)
    expected_var = 1.0 / (4 * (1.0 / 4.0 + 1.0))
    assert math.isclose(float(mu_post), expected_mu, rel_tol=1e-6)
    assert math.isclose(float(var_post), expected_var, rel_tol=1e-6)


def test_standardized_quadrature_for_sine():
    mu = 0.3
    var = 1.0e-8
    numerical = float(ref_im_sol(jnp.array(mu), jnp.array(var), observable_sin))
    exact = math.exp(-var / 2.0) * math.sin(mu)
    assert math.isclose(numerical, exact, rel_tol=2e-5, abs_tol=2e-6)


def test_without_replacement_full_batch_mean():
    y = jnp.arange(8.0)
    means = _sample_minibatch_means(
        jax.random.PRNGKey(0),
        y,
        s=8,
        n_experiments=5,
        denom=1.0,
        replace=False,
    )
    expected = float(jnp.mean(y))
    assert jnp.allclose(means, expected)


def test_end_to_end_costs_are_finite():
    result = costs(
        jax.random.PRNGKey(7),
        m=8,
        s=2,
        epsilon=0.25,
        sig_th=1.0,
        sig_y=1.0,
        func=observable_sin,
        n_experiments=6,
        n=1,
        replace=False,
        horizon_factor=0.2,
        max_validation_paths=16,
    )
    assert len(result) == 4
    assert all(math.isfinite(float(x)) and float(x) >= 0 for x in result)


def test_full_batch_sgld_matches_em():
    e_rmse, e_cost, s_rmse, s_cost = costs(
        jax.random.PRNGKey(9),
        m=8,
        s=8,
        epsilon=0.25,
        sig_th=1.0,
        sig_y=1.0,
        func=observable_sin,
        n_experiments=6,
        n=1,
        replace=False,
        horizon_factor=0.2,
        max_validation_paths=16,
    )
    assert float(e_cost) == float(s_cost)
    assert float(e_rmse) == float(s_rmse)
