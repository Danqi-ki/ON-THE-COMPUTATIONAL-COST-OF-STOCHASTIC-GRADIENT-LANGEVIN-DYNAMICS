# On the Computational Cost of Stochastic Gradient Langevin Dynamics

Reproducible numerical code for the empirical computational-cost comparison
between the full Euler--Maruyama (EM) discretisation and Stochastic Gradient
Langevin Dynamics (SGLD) in the accompanying manuscript by Mateusz B. Majka,
Tigran Nagapetyan, Yue Wu, and Danqi Zhuang.

The repository focuses on the **cost experiments**.  It implements the
Gaussian benchmark for which the posterior and invariant distribution are
known, calibrates a time step by a coupled step-doubling test, estimates the
Monte Carlo path count, and compares the leading operation-count costs of EM
and SGLD.

## Scientific convention used by the code

The benchmark data model is

```text
theta ~ N(0, sigma_theta^2 / m),
Y_i | theta ~ N(theta, sigma_y^2),    i=1,...,m.
```

The posterior is `N(mu_p, sigma_p^2)` with

```text
sigma_p^2 = (1/m) * (sigma_theta^{-2} + sigma_y^{-2})^{-1},
mu_p      = mean(Y) / (sigma_y^2 / sigma_theta^2 + 1).
```

The code simulates the **time-rescaled** Langevin dynamics

```text
dX_t = -(X_t - mu_p)/(2 m sigma_p^2) dt + m^{-1/2} dW_t.
```

Therefore the symbol `h` in the code is the step size for this rescaled SDE.
The rescaling keeps the Lipschitz/dissipativity constants independent of `m`
and makes the diffusion coefficient `m^{-1/2}`.

For SGLD, mini-batches are sampled **without replacement within each step** by
default, matching the manuscript.  Mini-batches are independent across time
steps and Monte Carlo paths.  `--with-replacement` is available only as a
sensitivity option.

When `s = m` and sampling is without replacement, SGLD is exactly the full EM
scheme.  The implementation enforces this identity rather than allowing pilot
Monte Carlo noise to create a spurious cost difference.

## Numerical cost protocol

For each `(m, s, epsilon)` configuration and each regenerated dataset:

1. Set `T = ceil(3 log(1/epsilon))` by default.
2. Start from `h = 4 epsilon`.
3. Generate coupled terminal outputs at `h` and `h/2`.  Brownian increments are
   nested; for SGLD the coarse path shares the first mini-batch in each pair of
   fine steps.
4. If the empirical mean squared level difference is larger than
   `epsilon^2 / 2`, replace `h` by `h/2` and repeat.
5. At the accepted **coarse** step size `h`, estimate the single-path variance
   `V` and take `N = max(1, ceil(2 V / epsilon^2))`.
6. Report the leading operation-count costs

```text
Cost(EM)   = N_EM   * ceil(T / h_EM)   * m,
Cost(SGLD) = N_SGLD * ceil(T / h_SGLD) * s.
```

These are operation-count proxies, not measured wall-clock times.  The result
files also contain wall-clock time for diagnostics.

The theoretical `N` can be very large.  By default the RMSE *diagnostic* is
estimated with at most 4096 paths, while the reported computational cost still
uses the full theoretical `N`.  Use `--max-validation-paths 0` to remove this
cap.  This distinction is important when interpreting the `*_rmse` columns.

## Important implementation details

- The posterior reference expectation is evaluated on a **standardised Gaussian
  grid** rather than a fixed `x` grid.  This remains accurate when `m` is large
  and the posterior variance is very small.
- Exact without-replacement mini-batches use Floyd's algorithm.  For batches
  close to the full data size, the code samples the smaller complement instead,
  so the sampler does not require an `O(m)` random permutation at every SGLD
  step.
- Every result file stores its complete configuration in the header.  A resumed
  run is rejected if the configuration does not match, preventing accidental
  mixing of results from different experiments.
- Parallel execution uses Python's `spawn` context, avoiding the common JAX +
  `fork` multiprocessing failure mode.  Use `--workers 1` on a GPU.

## Repository layout

```text
run.py                  command-line entry point
src/cost_funcs.py       model, coupled step-size search, EM/SGLD simulators
src/__init__.py
tests/test_smoke.py     posterior, quadrature, sampling, end-to-end tests
requirements.txt
requirements-dev.txt
.github/workflows/ci.yml
```

## Installation

Python 3.10--3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For development/tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The default requirements install the standard JAX package.  For NVIDIA GPU
execution, install a JAX build appropriate for your CUDA environment, for
example on a supported Linux/WSL2 setup:

```bash
pip install "jax[cuda12]"
```

See the official JAX installation instructions for the correct command for
your platform/CUDA version.

## First run: smoke test

Run this before launching a large experiment:

```bash
python run.py smoke --device cpu
```

A successful run ends with `Smoke test passed.`

## Cost winner maps in the `(m,s)` plane

The `sample` mode generates pseudo-random or Sobol `(m,s)` pairs for each fixed
accuracy level and produces both log-scale and linear-scale winner maps.

```bash
python run.py sample \
  --epsilons 1e-3 5e-4 \
  --n-points 500 \
  --sampling sobol \
  --m-min 2 \
  --m-max 1000 \
  --func sin \
  --n-experiments 25 \
  --n-rep 5 \
  --workers 1
```

The plots include the leading-order boundary

```text
s = m - epsilon m^2,
```

which is the EM/SGLD boundary in the small-data regime where that expression is
positive.  The empirical markers are determined by the simulated cost
estimates, not by the theoretical boundary.

## Fixed parameter grid / tables

Use `grid` to evaluate explicit parameter combinations.  Invalid combinations
with `s > m` are skipped; `s = m` is valid.

```bash
python run.py grid \
  --m-list 100 1000 100000 \
  --s-list 20 50 100 \
  --epsilons 1e-3 \
  --func sin \
  --n-experiments 50 \
  --n-rep 5
```

## Main command-line options

| option | default | meaning |
| --- | ---: | --- |
| `--sig-th` | `0.2` | `sigma_theta` in the scaled prior |
| `--sig-y` | `0.15` | observation-noise standard deviation |
| `--initial` | `1.0` | initial Langevin state |
| `--func` | `sin` | `sin`, `sin_abs`, or `square` |
| `--n-experiments` | `50` | pilot paths for step selection/variance |
| `--n-rep` | `5` | regenerated datasets per parameter point |
| `--horizon-factor` | `3` | `T = ceil(factor log(1/epsilon))` |
| `--max-validation-paths` | `4096` | cap used only for RMSE diagnostic; `0` = no cap |
| `--with-replacement` | off | use non-paper with-replacement mini-batches |
| `--workers` | `1` | parallel parameter points |
| `--device` | auto | optionally force `cpu` or require `gpu` |
| `--out-dir` | `results` | output directory |

Use `python run.py <mode> --help` for mode-specific options.

## Output and resuming

Each experiment produces:

- `*.txt`: one row per completed point, plus the exact JSON configuration in
  the header;
- `*.npz`: the same results in NumPy arrays, including the configuration;
- `*_log.png` and `*_linear.png`: winner maps for `sample` mode.

Rows are written immediately after each point finishes.  Re-running the same
command resumes missing points.  A short hash of the experiment configuration
is included in each filename, and the full configuration is checked before a
resume.

## Reproducibility checks

The test suite checks:

- the posterior formula for the scaled prior;
- reference quadrature at a very small posterior variance;
- exact full-batch behaviour of without-replacement sampling;
- an end-to-end finite cost calculation.

GitHub Actions runs the test suite and a CLI smoke test on every push and pull
request.

## Scope

The present repository is intended to reproduce and audit the **empirical cost
comparison**.  Auxiliary strong-convergence and variance figures from the
manuscript should be kept in separate, explicitly named scripts if they are
added later, so that their time-step conventions cannot be confused with the
cost experiment implemented here.
