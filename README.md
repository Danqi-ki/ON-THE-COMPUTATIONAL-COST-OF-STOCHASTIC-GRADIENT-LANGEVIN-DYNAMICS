# TrueCost

Empirical cost comparison between the Euler–Maruyama scheme and Stochastic
Gradient Langevin Dynamics (SGLD) for sampling from a Bayesian posterior over
a scalar mean `mu`, given `m` noisy observations `Y_i = mu + sig_y * eps_i`
with prior `mu ~ N(0, sig_th^2)`.

For each `(m, s, epsilon)` triple (`m` = number of observations, `s` =
SGLD minibatch size, `epsilon` = target RMSE), the code:

1. Searches for a stepsize `h` (via a coupled coarse/fine level-MSE
   criterion) that meets the accuracy target for both Euler and SGLD.
2. Estimates the number of Monte Carlo paths `N` needed from the
   stepsize-search variance.
3. Reports the leading computational cost (`N * n_steps * m` for Euler,
   `N * n_steps * s` for SGLD) and the empirical RMSE against a reference
   solution, computed by numerical integration against the exact posterior.

## Project layout

```
run.py                          CLI entry point (see below)
src/
  cost_funcs.py                 JAX backend: stepsize search, path
                                 simulation, and the `costs()` function
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of JAX. For GPU acceleration on
Windows, run inside WSL2 and install the CUDA build instead:

```bash
pip install "jax[cuda12]"
```

## Usage

`run.py` has two modes, selected as the first positional argument.

### `sample` — random/quasi-random (m, s) sweep

Draws `n_points` random `(m, s)` pairs and evaluates each one, producing a
result file, an `.npz` summary, and a log-log + linear-scale plot per
epsilon value.

```bash
python run.py sample --epsilons 1e-2 1e-3 --n-points 500 --workers 4
```

Key options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--epsilons` | `1e-1 1e-2 1e-3 1e-4 1e-5` | Target RMSE values to run |
| `--n-points` | `1000` | Number of `(m, s)` pairs per epsilon |
| `--sampling` | `uniform` | `uniform` (pseudo-random) or `sobol` (quasi-random) |
| `--m-min` / `--m-max` | `2` / `1000` | Range of `m` to sample from |
| `--plot-only` | off | Skip computation, just replot existing result files |

### `grid` — fixed (m, s, epsilon) grid

Evaluates every valid combination from explicit `m`, `s`, and `epsilon`
lists into a single result file (`s` values `>= m` for a given `m` are
skipped).

```bash
python run.py grid --m-list 1000 5000 --s-list 10 100 --epsilons 1e-2 1e-3
```

### Shared options (both modes)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--sig-th` | `0.5` | Prior std dev of `mu`: `mu ~ N(0, sig_th^2)` |
| `--sig-y` | `0.25` | Observation noise std dev: `Y_i = mu + sig_y * eps_i` |
| `--initial` | `1.0` | Initial state for the Euler/SGLD paths |
| `--func` | `square` | Functional `func(y)` applied to the terminal state and to the reference posterior integral. Choices: `square` (`y**2`), `sin` (`sin(y)`), `sin_abs` (`sin(y) + abs(y) + 1`) |
| `--n-experiments` | `100` | Pilot paths used for stepsize search / variance estimate |
| `--n-rep` | `10` | Independent repetitions per point (RMSE/cost are averaged) |
| `--seed` | `2026` | Base seed for reproducibility |
| `--workers` | `1` | Parallel worker processes across points |
| `--out-dir` | `.` | Output directory |
| `--device` | auto-detect | Force `cpu` or `gpu` |
| `--vectorized` | off | Vectorized stepsize search (GPU only) |

Both modes write results incrementally to a `.txt` file, so a run can be
interrupted and resumed by re-running the same command — completed points
are skipped and only the missing ones are recomputed.

## Output files

- `*.txt` — one row per completed `(m, s, epsilon)` point (point index, m,
  s, epsilon, Euler/SGLD cost, Euler/SGLD RMSE, winner, wall-clock seconds).
- `*.npz` — the same data as NumPy arrays, plus total wall-clock time.
- `*.png` (sample mode only) — log-log and linear-scale scatter plots of
  which method is cheaper across the sampled `(m, s)` region.
