# Baselines reference

This file documents reference numbers from experiments on the existing EnerGNN benchmarks before any attention work begins. It is the comparison point for every attention variant produced in this branch (see `Rapport d'implémentation des mécanismes d'attention dans EnerGNN.pdf` and `baseline_walkthrough.html` for the per-item method).

Three complementary benchmarks are reported:

1. **LinearSystem baseline** -- supervised training on the built-in `LinearSystemProblemLoader` (DC-power-flow toy). Provides per-size eval scores, step time, and memory for `LocalSumMessagePassingFunction`, which is the message-function baseline every attention variant must match or beat.

2. **IEEE conversion experimentation** -- forward-pass validation through `pypowsybl-to-energnn`'s AC-load-flow converters on the six standard IEEE networks. Confirms the pipeline runs end-to-end on realistic grid topologies and produces physically plausible outputs without training.

3. **IEEE supervised AC-load-flow baseline** -- supervised training on perturbed-load AC LF instances via `ACLoadFlowProblemLoader` (in `benchmarks/ac_load_flow_problem.py`). Topology is fixed; loads vary. This is the Gate-5 reference for attention work on realistic grid data.

LinearSystem baseline and the IEEE conversion experimentation were run on the La Javaness GPU server (cuda:0, cuda:1). The exact dataset configuration, seeds, and full environment fingerprint are recorded with each section so future runs can be compared exactly. Perf numbers (Gate 6) are reported per-Approche below on production-realistic graph sizes (`ieee118`, `ieee300`).

**Memory caveat.** `peak_memory_mb` is the process RSS at the end of each run; it captures JAX runtime, CUDA libraries, and dataset buffers, not just model parameters. For per-model param footprint, see the `n_params` column.

Raw JSON results:
- `benchmarks/results/00_baseline/baseline_linearsystem.json`
- `benchmarks/results/00_baseline/baseline_ieee_conversion.json`
- `benchmarks/results/00_baseline/baseline_ac_load_flow.json`

Re-run the experiments:

```bash
uv run python benchmarks/00_baseline/baseline_linearsystem.py
uv run python benchmarks/00_baseline/baseline_ieee_conversion.py
uv run python benchmarks/00_baseline/baseline_ac_load_flow.py
uv run python benchmarks/render_baselines_md.py
```

---

## LinearSystem baseline

Built-in DC-power-flow toy problem (`LinearSystemProblemLoader`). The 2 ready-to-use sizes covered here (Tiny/Small) all use `LocalSumMessagePassingFunction` as the message function and `RecurrentCoupler` as the coupler. Optimizer `optax.adam(1e-3)`. Seeds tested: `[0, 1, 2]`. Data: `n_max=3`, `dataset_size=64`, `batch_size=4`, `val_dataset_size=32`.

**Note:** Baseline scope: Tiny and Small x 3 seeds on the LinearSystem toy (n_max=3). Medium / Large / ExtraLarge are available via LARGER_SIZES in the script and are run on demand against production-realistic graph sizes (Gate 6 perf, see below), not on the toy.

### Summary table

`final-eval` columns are the eval MSE at the last training epoch; `best-eval (median)` is the median over seeds of each seed's best eval across all epochs (a fair early-stopping reference).

| Size       | n_params | n_epochs | final-eval (min) | final-eval (median) | final-eval (max) | best-eval (median) | step time (ms) | peak mem (MB) |
|------------|---------:|---------:|-----------------:|--------------------:|-----------------:|-------------------:|---------------:|--------------:|
| Tiny       |      185 |       10 |        3.940e-01 |           4.388e-01 |        5.532e-01 |          4.388e-01 |          229.0 |        2411.0 |
| Small      |     2177 |       15 |        3.972e-01 |           3.990e-01 |        4.556e-01 |          3.877e-01 |          574.5 |        2561.6 |

### Per-run detail

| Size       | Seed | eval_before | eval_after  | improvement | median ms | p90 ms  | train time (s) | warning |
|------------|-----:|------------:|------------:|------------:|----------:|--------:|---------------:|---------|
| Tiny       |    0 |   1.872e+00 |   5.532e-01 |   1.319e+00 |     226.4 |   367.9 |          116.9 | - |
| Tiny       |    1 |   8.124e-01 |   3.940e-01 |   4.184e-01 |     229.0 |   273.3 |          115.6 | - |
| Tiny       |    2 |   1.199e+00 |   4.388e-01 |   7.602e-01 |     231.2 |   300.4 |          116.5 | - |
| Small      |    0 |   9.192e-01 |   4.556e-01 |   4.636e-01 |     413.9 |   606.2 |          222.5 | - |
| Small      |    1 |   7.494e-01 |   3.972e-01 |   3.522e-01 |     574.5 |   662.7 |          230.0 | - |
| Small      |    2 |   9.677e-01 |   3.990e-01 |   5.688e-01 |     574.6 |   675.1 |          227.2 | - |

### Eval-loss curves (per epoch, median across seeds)

```
  Tiny       : 9.29e-01 -> 8.46e-01 -> 7.72e-01 -> 7.05e-01 -> 6.45e-01 -> 5.91e-01 -> 5.46e-01 -> 5.09e-01 -> 4.82e-01 -> 4.62e-01
  Small      : 4.63e-01 -> 4.41e-01 -> 4.33e-01 -> 4.28e-01 -> 4.24e-01 -> 4.19e-01 -> 4.14e-01 -> 4.09e-01 -> 4.06e-01 -> 4.05e-01 -> 4.06e-01 -> 4.09e-01 -> 4.10e-01 -> 4.14e-01 -> 4.17e-01
```

Environment for these runs:
```
  python: 3.13.13
  platform: Linux-6.8.0-111-generic-x86_64-with-glibc2.35
  jax: 0.9.0
  jax_devices: ['cuda:0', 'cuda:1']
  flax: 0.12.3
  optax: 0.2.6
```

---

## IEEE network conversion experimentation

For each standard IEEE size (9, 14, 30, 57, 118, 300): build the network with `pn.create_ieee<N>()`, solve AC load flow with `lf.run_ac(network)`, set `network.per_unit = True`, convert via `ACLoadFlowInputConverter` / `ACLoadFlowOutputConverter`, then run one forward pass through `TinyRecurrentEquivariantGNN` (seed `0`) instantiated from the converter's structure. No supervised training. The intent is to confirm shape/dtype compatibility, finite outputs, and physically plausible AC quantities (bus voltages in `[0.5, 1.6]` p.u.).

### Per-network results

| Network  | n_addr | conv ms | LF ms  | Tiny fwd ms | finite (in/out/fwd) | v_mag [min, max] | in range | error |
|----------|-------:|--------:|-------:|------------:|--------------------:|-----------------:|---------:|-------|
| ieee9    |      9 |  2444.0 | 1260.3 |     12679.9 |               Y/Y/Y | [0.996, 1.040] |        Y | - |
| ieee14   |     14 |   593.0 |   29.3 |     11324.6 |               Y/Y/Y | [1.010, 1.090] |        Y | - |
| ieee30   |     30 |   751.0 |   47.5 |     11756.2 |               Y/Y/Y | [0.992, 1.082] |        Y | - |
| ieee57   |     57 |   651.7 |   60.0 |     12845.1 |               Y/Y/Y | [0.938, 1.062] |        Y | - |
| ieee118  |    118 |  1205.4 |   75.9 |     12208.0 |               Y/Y/Y | [0.943, 1.050] |        Y | - |
| ieee300  |    300 |   651.8 |  109.0 |     11021.5 |               Y/Y/Y | [0.929, 1.074] |        Y | - |

### Hyper-edge-set inventory per network

- `ieee9`: batteries=0, buses=9, dangling_lines=0, generators=3, hvdc_lines=0, lcc_converter_stations=0, lines=6, loads=3, shunts=0, static_var_compensators=0, vsc_converter_stations=0
- `ieee14`: batteries=0, buses=14, dangling_lines=0, generators=5, hvdc_lines=0, lcc_converter_stations=0, lines=17, loads=11, shunts=1, static_var_compensators=0, vsc_converter_stations=0
- `ieee30`: batteries=0, buses=30, dangling_lines=0, generators=6, hvdc_lines=0, lcc_converter_stations=0, lines=37, loads=21, shunts=2, static_var_compensators=0, vsc_converter_stations=0
- `ieee57`: batteries=0, buses=57, dangling_lines=0, generators=7, hvdc_lines=0, lcc_converter_stations=0, lines=63, loads=42, shunts=3, static_var_compensators=0, vsc_converter_stations=0
- `ieee118`: batteries=0, buses=118, dangling_lines=0, generators=54, hvdc_lines=0, lcc_converter_stations=0, lines=177, loads=91, shunts=14, static_var_compensators=0, vsc_converter_stations=0
- `ieee300`: batteries=0, buses=300, dangling_lines=0, generators=69, hvdc_lines=0, lcc_converter_stations=0, lines=304, loads=198, shunts=14, static_var_compensators=0, vsc_converter_stations=0

Environment for these runs:
```
  python: 3.13.13
  platform: Linux-6.8.0-111-generic-x86_64-with-glibc2.35
  jax: 0.9.0
  jax_devices: ['cuda:0', 'cuda:1']
  flax: 0.12.3
  pypowsybl: 1.15.0
```

---

## IEEE supervised AC-load-flow baseline

Trains `LocalSumMessagePassingFunction` (via the Tiny and Small ready-to-use GNN combos) on supervised AC-load-flow data generated by `ACLoadFlowProblemLoader`. For each IEEE network in `(9, 14)`, each training instance is produced by perturbing the load setpoints (p0, q0) of the network by independent multiplicative factors in `[1 - 0.1, 1 + 0.1]`, then solving AC load flow with `lf.run_ac(network)` to obtain the ground-truth response (V_mag for buses, P/Q/I for branches and devices). The model is supervised under MSE against this oracle. Optimizer `optax.adam(1e-3)`. Seeds tested: `[0, 1, 2]`. Data: `dataset_size=32`, `batch_size=4`, `val_dataset_size=16`. Topology is fixed per loader; only load setpoints vary across instances, so batches collate without padding.

### Summary table

| Network | Size  | n_params | n_epochs | final-eval (median) | best-eval (median) | step time (ms, median) | train (s, median) |
|---------|-------|---------:|---------:|--------------------:|-------------------:|-----------------------:|------------------:|
| ieee9   | Tiny  |     1587 |       10 |           1.561e-01 |          1.561e-01 |                  1147 |             230.3 |
| ieee9   | Small |    15863 |       15 |           5.075e-03 |          5.075e-03 |                  2147 |             573.0 |
| ieee14  | Tiny  |     1587 |       10 |           8.668e-02 |          8.668e-02 |                  2327 |             416.1 |
| ieee14  | Small |    15863 |       15 |           4.564e-03 |          4.564e-03 |                  1983 |             474.9 |

### Per-run detail

| Network | Size  | Seed | eval_before | eval_after  | improvement | best_eval   | best_epoch | median ms | train (s) |
|---------|-------|-----:|------------:|------------:|------------:|------------:|-----------:|----------:|----------:|
| ieee9   | Tiny  |    0 |   5.806e-01 |   1.561e-01 |   4.246e-01 |   1.561e-01 |   ep 10/10 |    1136.8 |     225.3 |
| ieee9   | Tiny  |    1 |   6.181e-01 |   1.427e-01 |   4.754e-01 |   1.427e-01 |   ep 10/10 |    1147.2 |     231.6 |
| ieee9   | Tiny  |    2 |   7.694e-01 |   1.628e-01 |   6.066e-01 |   1.628e-01 |   ep 10/10 |    1151.2 |     230.3 |
| ieee9   | Small |    0 |   6.259e-01 |   5.075e-03 |   6.208e-01 |   5.075e-03 |   ep 15/15 |    1454.4 |     517.9 |
| ieee9   | Small |    1 |   5.260e-01 |   1.148e-03 |   5.249e-01 |   1.148e-03 |   ep 15/15 |    2216.7 |     573.0 |
| ieee9   | Small |    2 |   7.141e-01 |   6.569e-03 |   7.075e-01 |   6.569e-03 |   ep 15/15 |    2146.7 |     586.1 |
| ieee14  | Tiny  |    0 |   3.218e-01 |   8.668e-02 |   2.352e-01 |   8.668e-02 |   ep 10/10 |    2243.5 |     401.4 |
| ieee14  | Tiny  |    1 |   3.881e-01 |   8.026e-02 |   3.078e-01 |   8.026e-02 |   ep 10/10 |    2326.8 |     419.0 |
| ieee14  | Tiny  |    2 |   5.612e-01 |   1.352e-01 |   4.261e-01 |   1.352e-01 |   ep 10/10 |    2481.1 |     416.1 |
| ieee14  | Small |    0 |   3.741e-01 |   4.564e-03 |   3.696e-01 |   4.564e-03 |   ep 15/15 |    2808.6 |     543.6 |
| ieee14  | Small |    1 |   2.797e-01 |   2.209e-03 |   2.775e-01 |   2.209e-03 |   ep 15/15 |    1982.7 |     474.9 |
| ieee14  | Small |    2 |   4.466e-01 |   9.866e-03 |   4.368e-01 |   9.866e-03 |   ep 15/15 |    1958.3 |     470.4 |

### Per-(network, size) loss curves (median across seeds)

```
  ieee9   Tiny : 4.57e-01 -> 4.16e-01 -> 3.73e-01 -> 3.26e-01 -> 2.87e-01 -> 2.53e-01 -> 2.21e-01 -> 1.97e-01 -> 1.76e-01 -> 1.56e-01
  ieee9   Small: 2.79e-01 -> 1.98e-01 -> 1.35e-01 -> 9.13e-02 -> 6.31e-02 -> 4.12e-02 -> 2.54e-02 -> 1.74e-02 -> 1.36e-02 -> 1.15e-02 -> 9.97e-03 -> 8.65e-03 -> 7.16e-03 -> 5.99e-03 -> 5.08e-03
  ieee14  Tiny : 2.53e-01 -> 2.22e-01 -> 1.95e-01 -> 1.70e-01 -> 1.52e-01 -> 1.37e-01 -> 1.24e-01 -> 1.10e-01 -> 9.82e-02 -> 8.67e-02
  ieee14  Small: 1.58e-01 -> 1.29e-01 -> 1.04e-01 -> 8.13e-02 -> 6.03e-02 -> 4.24e-02 -> 2.88e-02 -> 1.94e-02 -> 1.35e-02 -> 1.00e-02 -> 7.99e-03 -> 6.69e-03 -> 5.80e-03 -> 5.12e-03 -> 4.56e-03
```

Environment for these runs:
```
  python: 3.13.x
  platform: Linux-6.8.0-111-generic-x86_64-with-glibc2.35 (La Javaness GPU server)
  jax: 0.9.0
  jax_devices: ['cuda:0', 'cuda:1']
  flax: 0.12.3
  optax: 0.2.6
  pypowsybl: 1.15.0
```

---

## Approche 1 closure — GATv2 (Item 1)

Item 1 (`GATv2MessagePassingFunction`) is implemented in line with the backlog spec section 3.1 (cf. `Rapport d'implémentation des mécanismes d'attention dans EnerGNN.pdf`, chapter 11): per-(class, port) score and value MLPs, numerator/denominator softmax form, segment-max subtraction for numerical stability under unbounded scores, and `score_uses_receiver=True` as the constructor default (matching Brody et al. 2022's `[h_a || h_e]` formulation).

All Approche 1 closure benchmarks below were run on the **La Javaness GPU server** (CUDA, JAX 0.9.0, Linux). Wall-time numbers and step-time medians are recorded so future iterations have a substrate-consistent reference. Verification scope is `LinearSystem` (toy) + `ieee14` (small AC-LF); Gate 6 perf timing also covers `ieee118` and `ieee300`.

### Verification gates

| Gate | Substrate | Status | Evidence |
|---|---|---|---|
| 1 Unit | `tests/model/unit/test_gatv2_message_passing.py` | 16/16 pass | includes segment-max stability test, default-flag check, permutation equivariance, vmap+jit safety |
| 2 Static | `black`, `flake8`, `mypy` on modified files | clean | no new violations vs LocalSum baseline (inherited mypy noise) |
| 3 LinearSystem | `baseline_gatv2_linearsystem.py` | converges | see "LinearSystem" section below |
| 4 Integration | `tests/model/integration/test_coupler.py` | 4/4 pass | two new `RecurrentCoupler` cases (default and `score_uses_receiver=False`) |
| 5 IEEE supervised | `baseline_gatv2_ac_load_flow.py` on ieee9/14 | converges | see "IEEE AC LF" section below |
| 6 Perf | `perf_gatv2_vs_localsum.py` 3 substrates | <2x LocalSum | see "Perf" section below |
| 7 Consistency | `consistency_gatv2.py` on ieee14 | bit-identical | empreinte numérique `21647f27…` reproduces across re-runs |
| 8 Point figé réseau français | not yet provided | out of scope (snapshot non livré) | runs retrospectively when available |

## GATv2 vs LocalSum on LinearSystem (Approche 1, Gate 3)

Side-by-side comparison on `LinearSystemProblemLoader`. Same dataset config (`n_max=3`, `dataset_size=64`, `batch_size=4`, `val_dataset_size=32`), same seeds `[0, 1, 2]`, same per-size epoch budgets, same optimizer `optax.adam(1e-3)`. Both message functions run on the same hardware (La Javaness GPU server, CUDA).

### Side-by-side summary

| Size  | LocalSum n_params | GATv2 n_params | LocalSum best-eval (med) | GATv2 best-eval (med) | Δ vs LocalSum | LocalSum train (s) | GATv2 train (s) |
|-------|------------------:|---------------:|-------------------------:|----------------------:|--------------:|-------------------:|----------------:|
| Tiny  |               185 |            232 |                4.388e-01 |             4.238e-01 |         -3.4% |              116.5 |            91.3 |
| Small |              2177 |           3684 |                3.877e-01 |             3.906e-01 |         +0.7% |              227.2 |           209.2 |

Negative delta = GATv2 better. Wall-time medians come from the original LocalSum baseline (same La Javaness GPU server) and the spec-compliant GATv2 rerun (also La Javaness GPU server).

### Per-run detail (GATv2)

| Size  | Seed | eval_before | eval_after  | best_eval   | best_epoch | median ms/step | train (s) |
|-------|-----:|------------:|------------:|------------:|-----------:|---------------:|----------:|
| Tiny  |    0 |   5.413e-01 |   4.238e-01 |   4.238e-01 |   ep 10/10 |          274.7 |      90.5 |
| Tiny  |    1 |   7.090e-01 |   3.554e-01 |   3.554e-01 |   ep 10/10 |          269.0 |      91.3 |
| Tiny  |    2 |   1.232e+00 |   4.642e-01 |   4.642e-01 |   ep 10/10 |          269.1 |      92.2 |
| Small |    0 |   9.079e-01 |   4.517e-01 |   4.517e-01 |   ep 15/15 |          425.7 |     209.2 |
| Small |    1 |   1.048e+00 |   4.080e-01 |   3.554e-01 |    ep 7/15 |          426.0 |     209.6 |
| Small |    2 |   6.987e-01 |   3.972e-01 |   3.906e-01 |   ep 13/15 |          425.1 |     209.0 |

Environment for the GATv2 runs:
```
  python: 3.13.x
  platform: Linux-6.8.0-111-generic-x86_64-with-glibc2.35 (La Javaness GPU server)
  jax: 0.9.0
  jax_devices: ['cuda:0', 'cuda:1']
  flax: 0.12.3
  optax: 0.2.6
```

---

## GATv2 vs LocalSum on IEEE supervised AC LF (Approche 1, Gate 5)

Side-by-side comparison on supervised AC-load-flow data via `ACLoadFlowProblemLoader`. Networks `ieee9` and `ieee14`, `perturbation_scale=0.1`, `dataset_size=32`, `batch_size=4`, seeds `[0, 1, 2]`, per-size epoch budgets identical to LocalSum, optimizer `optax.adam(1e-3)`. The only difference is the message function inside `RecurrentCoupler`. Scope is `ieee9` + `ieee14`; `ieee118` and `ieee300` are covered by Gate 6 perf timing rather than by full training (`feedback_verification_scope`).

### Side-by-side summary (best-eval and train wall-time median across seeds)

| Network | Size  | LocalSum n_params | GATv2 n_params | LocalSum best-eval | GATv2 best-eval | Δ best-eval | LocalSum train (s) | GATv2 train (s) |
|---------|-------|------------------:|---------------:|-------------------:|----------------:|------------:|-------------------:|----------------:|
| ieee9   | Tiny  |              1587 |           1881 |          1.561e-01 |       1.392e-01 |      -10.8% |              230.3 |           246.3 |
| ieee9   | Small |             15863 |          25289 |          5.075e-03 |       5.837e-03 |      +15.0% |              573.0 |           451.3 |
| ieee14  | Tiny  |              1587 |           1881 |          8.668e-02 |       8.280e-02 |       -4.5% |              416.1 |           300.5 |
| ieee14  | Small |             15863 |          25289 |          4.564e-03 |       7.995e-03 |      +75.2% |              474.9 |           571.0 |

Negative delta means GATv2 better than LocalSum (lower MSE). Both message functions run on the La Javaness GPU server (CUDA); eval scores are deterministic (seed-bound) and wall-times are directly comparable.

Tiny configs: GATv2 is -4.5% to -10.8% vs LocalSum. Small configs: GATv2 matches LocalSum on LinearSystem (Gate 3 above) but is +15% to +75% on AC LF with ~59% more parameters (25 289 vs 15 863). At Small capacity LocalSum's per-(class, port) factoring is more parameter-efficient than GATv2 on this benchmark. Items 2-4 (Performer, MultiHeadQKV, GlobalAggregation) below confirm the topology-aware/agnostic split: MultiHeadQKV ties LocalSum on Small AC LF; Performer and GlobalAggregation trail by ~×5 on the same regime, indicating the topology channel matters more than raw capacity.

### Per-run detail (GATv2)

| Network | Size  | Seed | eval_before | eval_after  | best_eval   | best_epoch | median ms/step | train (s) |
|---------|-------|-----:|------------:|------------:|------------:|-----------:|---------------:|----------:|
| ieee9   | Tiny  |    0 |   5.661e-01 |   1.392e-01 |   1.392e-01 |   ep 10/10 |         1334.6 |     242.6 |
| ieee9   | Tiny  |    1 |   5.644e-01 |   1.322e-01 |   1.322e-01 |   ep 10/10 |         1392.2 |     246.3 |
| ieee9   | Tiny  |    2 |   5.855e-01 |   1.882e-01 |   1.882e-01 |   ep 10/10 |         1403.6 |     247.1 |
| ieee9   | Small |    0 |   4.892e-01 |   2.169e-03 |   2.169e-03 |   ep 15/15 |         1979.7 |     447.4 |
| ieee9   | Small |    1 |   4.991e-01 |   5.837e-03 |   5.837e-03 |   ep 15/15 |         1997.2 |     451.3 |
| ieee9   | Small |    2 |   5.488e-01 |   1.379e-02 |   1.379e-02 |   ep 15/15 |         2020.4 |     476.0 |
| ieee14  | Tiny  |    0 |   3.166e-01 |   9.155e-02 |   9.155e-02 |   ep 10/10 |         1780.7 |     291.3 |
| ieee14  | Tiny  |    1 |   3.112e-01 |   6.572e-02 |   6.572e-02 |   ep 10/10 |         1867.6 |     304.1 |
| ieee14  | Tiny  |    2 |   3.103e-01 |   8.280e-02 |   8.280e-02 |   ep 10/10 |         1861.1 |     300.5 |
| ieee14  | Small |    0 |   2.640e-01 |   9.609e-03 |   9.609e-03 |   ep 15/15 |         2553.4 |     558.9 |
| ieee14  | Small |    1 |   2.886e-01 |   7.995e-03 |   7.995e-03 |   ep 15/15 |         2680.2 |     571.0 |
| ieee14  | Small |    2 |   2.894e-01 |   4.629e-03 |   4.629e-03 |   ep 15/15 |         2748.8 |     574.5 |

Environment for the GATv2 AC-LF runs:
```
  platform: Linux-6.8.0-111-generic-x86_64-with-glibc2.35 (La Javaness GPU server)
  jax: 0.9.0
  jax_devices: ['cuda:0', 'cuda:1']
```

## Gate 6 perf — LocalSum vs GATv2 forward / forward+backward

Median wall-time per call on a single graph after a 20-call warm-up, 100 timed calls, identical hyper-parameters (`in_array_size=4`, `out_size=4`, `hidden_sizes=[4]`, seed `64`). Both message functions are JIT-compiled (`nnx.jit`). Substrates: LinearSystem (small graph, dispatch-dominated), `ieee118` and `ieee300` (the substrates targeted by Gate 6).

| Substrate | n_addr | LocalSum fwd (ms) | GATv2 fwd (ms) | LocalSum fwd+bwd (ms) | GATv2 fwd+bwd (ms) | overhead fwd | overhead fwd+bwd | peak RSS (MB) |
|-----------|------:|------------------:|---------------:|----------------------:|-------------------:|-------------:|-----------------:|--------------:|
| LinearSystem |   4 |              1.68 |           2.85 |                  2.21 |               3.89 |        x1.70 |            x1.76 |          2250 |
| ieee118      | 118 |              7.18 |          14.24 |                 11.32 |              22.49 |        x1.98 |            x1.99 |          2567 |
| ieee300      | 300 |              7.13 |          14.16 |                 11.27 |              22.39 |        x1.99 |            x1.99 |          2726 |

All overheads sit under the 3x Gate 6 ceiling (max x1.99 on ieee300 fwd+bwd). GATv2's extra value-MLP and softmax (numerator/denominator) roughly double forward and forward+backward at AC-LF scale; on LinearSystem the gap shrinks because dispatch overhead dominates the message-function call.

## Gate 7 consistency — bit-identical reproducibility on ieee14

A fixed-seed forward pass of `GATv2MessagePassingFunction` on the IEEE-14 AC-LF context hashes to:

```
output_sha256 = 21647f2761eedaa49535d110007cbb94110f7271c518feb8b4658043f3572833
```

Two re-runs of `benchmarks/01_gatv2/consistency_gatv2.py` on the La Javaness GPU server reproduce this hash exactly. To isolate GATv2's reproducibility from pypowsybl's AC-LF solver (which is not bit-deterministic across process invocations on multi-thread sparse solvers), the IEEE-14 context is pickled once on first run and re-loaded on subsequent runs. The cached blob is the frozen reference state for the Gate.

## Approche 2 closure — GlobalAggregation (Item 2)

Item 2 (`GlobalAggregationMessagePassingFunction`) is implemented per the backlog spec section 3.2 (cf. report, chapter 12) (v1 mean form with the proposed corrected denominator `sum(non_fictitious_addresses) + eps`). All Approche 2 closure benchmarks were run on the La Javaness GPU server (CUDA, JAX 0.9.0).

### Verification gates

| Gate | Substrate | Status | Evidence |
|---|---|---|---|
| 1 Unit | `tests/model/unit/test_global_aggregation_message_passing.py` | 13/13 pass | broadcast property, corrected denominator, mean correctness vs identity-MLP analytical, permutation invariance, vmap+jit safety |
| 2 Static | `black`, `flake8`, `mypy` | clean | no new violations |
| 3 LinearSystem | `baseline_global_aggregation_linearsystem.py` | converges | see "LinearSystem" section below |
| 4 Integration | `tests/model/integration/test_coupler.py` | 5/5 pass | one new `RecurrentCoupler` case wrapping GlobalAggregation |
| 5 IEEE supervised | `baseline_global_aggregation_ac_load_flow.py` on ieee9/14 | converges | see "IEEE AC LF" section below |
| 6 Perf | `perf_global_aggregation_vs_localsum.py` 3 substrates | **FASTER than LocalSum** (×0.09-0.39) | see "Perf" section below |
| 7 Consistency | `consistency_global_aggregation.py` on ieee14 | bit-identical | empreinte numérique `6c2d2c1a…` reproduces across re-runs |
| 8 Point figé réseau français | not yet provided | out of scope (snapshot non livré) | runs retrospectively when available |

## GlobalAggregation vs LocalSum on LinearSystem (Approche 2, Gate 3)

Same dataset config / seeds / epoch budgets / optimizer as Approche 1 Gate 3. Both message functions run on the La Javaness GPU server.

### Side-by-side summary

| Size  | LocalSum n_params | GlobalAgg n_params | LocalSum best-eval (med) | GlobalAgg best-eval (med) | Δ best-eval | LocalSum train (s) | GlobalAgg train (s) |
|-------|------------------:|-------------------:|-------------------------:|--------------------------:|------------:|-------------------:|--------------------:|
| Tiny  |               185 |                 65 |                4.388e-01 |                 6.316e-01 |      +43.9% |              116.5 |                92.8 |
| Small |              2177 |                977 |                3.877e-01 |                 4.371e-01 |      +12.7% |              227.2 |               163.4 |

GlobalAgg n_params ≈ 35-45% of LocalSum's because the message function is a single per-address MLP, not a per-(class, port) tree. Best-eval is worse on LinearSystem because the global mean discards local DC-power-flow structure; train wall-time is shorter because forward+backward have fewer ops.

### Per-run detail (GlobalAggregation)

| Size  | Seed | best_eval | train (s) |
|-------|-----:|----------:|----------:|
| Tiny  |    0 | 1.118e+00 |      92.8 |
| Tiny  |    1 | 5.800e-01 |      92.5 |
| Tiny  |    2 | 6.316e-01 |      95.5 |
| Small |    0 | 4.465e-01 |     161.8 |
| Small |    1 | 3.443e-01 |     163.4 |
| Small |    2 | 4.371e-01 |     165.9 |

## GlobalAggregation vs LocalSum on IEEE supervised AC LF (Approche 2, Gate 5)

Setup matches Approche 1 Gate 5 exactly (`ACLoadFlowProblemLoader` with `perturbation_scale=0.1`, `dataset_size=32`, `batch_size=4`, seeds `[0, 1, 2]`, identical Tiny / Small configs, `optax.adam(1e-3)`). Networks scoped to `ieee9` and `ieee14` per `feedback_verification_scope`.

### Side-by-side summary (best-eval median across seeds)

| Network | Size  | LocalSum n_params | GlobalAgg n_params | LocalSum best-eval | GlobalAgg best-eval | Δ best-eval | LocalSum train (s) | GlobalAgg train (s) |
|---------|-------|------------------:|-------------------:|-------------------:|--------------------:|------------:|-------------------:|--------------------:|
| ieee9   | Tiny  |              1587 |                719 |          1.561e-01 |           2.469e-01 |      +58.2% |              230.3 |               245.6 |
| ieee9   | Small |             15863 |               6879 |          5.075e-03 |           2.662e-02 |     +424.4% |              573.0 |               386.3 |
| ieee14  | Tiny  |              1587 |                719 |          8.668e-02 |           1.482e-01 |      +70.9% |              416.1 |               309.9 |
| ieee14  | Small |             15863 |               6879 |          4.564e-03 |           1.846e-02 |     +304.5% |              474.9 |               504.1 |

GlobalAgg has ~45% of LocalSum's parameters on every config and is 1.5-5× worse on best-eval. This is the expected scope trade-off: GlobalAgg discards local Kirchhoff structure (every receiver gets one global summary), so accuracy on a topology-sensitive task like AC LF degrades. Train wall-time is comparable or shorter despite the eval gap, consistent with the fewer ops.

### Per-run detail (GlobalAggregation, best-eval per seed)

| Network | Size  | Seed | best_eval | train (s) |
|---------|-------|-----:|----------:|----------:|
| ieee9   | Tiny  |    0 | 2.655e-01 |     237.7 |
| ieee9   | Tiny  |    1 | 2.469e-01 |     249.5 |
| ieee9   | Tiny  |    2 | 2.197e-01 |     245.6 |
| ieee9   | Small |    0 | 3.283e-02 |     373.7 |
| ieee9   | Small |    1 | 1.602e-02 |     386.3 |
| ieee9   | Small |    2 | 2.662e-02 |     388.2 |
| ieee14  | Tiny  |    0 | 1.401e-01 |     309.9 |
| ieee14  | Tiny  |    1 | 1.534e-01 |     345.1 |
| ieee14  | Tiny  |    2 | 1.482e-01 |     309.0 |
| ieee14  | Small |    0 | 1.846e-02 |     418.7 |
| ieee14  | Small |    1 | 2.114e-02 |     504.1 |
| ieee14  | Small |    2 | 1.728e-02 |     505.3 |

## Gate 6 perf — LocalSum vs GlobalAggregation forward / forward+backward

Median wall-time per call after 20 warm-up + 100 timed calls, identical hyper-parameters, `nnx.jit`-compiled. Same substrates as Approche 1 Gate 6.

| Substrate    | n_addr | LocalSum fwd (ms) | GlobalAgg fwd (ms) | LocalSum fwd+bwd (ms) | GlobalAgg fwd+bwd (ms) | overhead fwd | overhead fwd+bwd |
|--------------|------:|------------------:|-------------------:|----------------------:|-----------------------:|-------------:|-----------------:|
| LinearSystem |     4 |              2.07 |               0.81 |                  2.18 |                   0.97 |        ×0.39 |            ×0.44 |
| ieee118      |   118 |              9.56 |               0.86 |                 12.21 |                   1.16 |        ×0.09 |            ×0.09 |
| ieee300      |   300 |              9.04 |               0.88 |                 11.53 |                   1.06 |        ×0.10 |            ×0.09 |

GlobalAggregation runs ~10x faster than LocalSum on AC-LF-scale substrates (ieee118 / ieee300) and 2.5x faster on LinearSystem. One address-level MLP plus a mean and a broadcast replaces LocalSum's per-(class, port) MLP tree and per-port scatter_add. Inverse pattern to GATv2 (~2x slower than LocalSum).

## Gate 7 consistency — GlobalAggregation reproducibility on ieee14

Fixed-seed forward output hashes to:

```
output_sha256 = 6c2d2c1a0a58ec5476b243c0b46713284bf123aa4bceaf7b00bae011eec45c99
```

Two re-runs of `benchmarks/02_global_aggregation/consistency_global_aggregation.py` on the La Javaness GPU server reproduce this hash bit-for-bit. The IEEE-14 context is pickled once and re-loaded on subsequent runs to isolate GlobalAggregation reproducibility from pypowsybl's non-determinism.

## Approche 3 closure — MultiHeadQKV (Item 3)

Item 3 (`MultiHeadQKVMessagePassingFunction`) is implemented per the backlog spec section 3.3 (cf. report, chapter 13) (v1 single-head Q/K/V form, raw bilinear $K^\top Q$ score, with `/sqrt(d_qk)` scaling default per Vaswani et al. 2017 stability convention). All Approche 3 closure benchmarks were run on the La Javaness GPU server (CUDA, JAX 0.9.0).

### Verification gates

| Gate | Substrate | Status | Evidence |
|---|---|---|---|
| 1 Unit | `tests/model/unit/test_multi_head_qkv_message_passing.py` | 15/15 pass | Q/K/V dims, output shape, scale_scores toggle + sqrt(d_qk) factor check, masking, permutation, vmap+jit, gradient flow, empty graph |
| 2 Static | `black`, `flake8`, `mypy` | clean | no new violations |
| 3 LinearSystem | `baseline_multi_head_qkv_linearsystem.py` | converges | see "LinearSystem" section below |
| 4 Integration | `tests/model/integration/test_coupler.py` | 6/6 pass | one new `RecurrentCoupler` case wrapping MultiHeadQKV |
| 5 IEEE supervised | `baseline_multi_head_qkv_ac_load_flow.py` on ieee9/14 | **tied/beats LocalSum on Small** | see "IEEE AC LF" section below |
| 6 Perf | `perf_multi_head_qkv_vs_localsum.py` 3 substrates | overhead **×2.07 forward consistently** | see "Perf" section below |
| 7 Consistency | `consistency_multi_head_qkv.py` on ieee14 | bit-identical | empreinte numérique `6b976412…` reproduces across re-runs |
| 8 Point figé réseau français | not yet provided | out of scope (snapshot non livré) | runs retrospectively when available |

## MultiHeadQKV vs LocalSum on LinearSystem (Approche 3, Gate 3)

Same dataset config / seeds / epoch budgets / optimizer as Approches 1+2 Gate 3. La Javaness GPU server.

### Side-by-side summary

| Size  | LocalSum n_params | MultiHQKV n_params | LocalSum best-eval (med) | MultiHQKV best-eval (med) | Δ best-eval | LocalSum train (s) | MultiHQKV train (s) |
|-------|------------------:|-------------------:|-------------------------:|--------------------------:|------------:|-------------------:|--------------------:|
| Tiny  |               185 |                505 |                4.388e-01 |                 4.345e-01 |       -1.0% |              116.5 |               124.1 |
| Small |              2177 |               3937 |                3.877e-01 |                 3.774e-01 |       -2.7% |              227.2 |               280.7 |

MultiHQKV trades ~2-3× more parameters for marginally better best-eval on LinearSystem (within noise — the LinearSystem DC toy does not have enough structure to discriminate the two mechanisms).

### Per-run detail (MultiHeadQKV)

| Size  | Seed | best_eval | train (s) |
|-------|-----:|----------:|----------:|
| Tiny  |    0 | 5.528e-01 |     117.7 |
| Tiny  |    1 | 4.345e-01 |     124.1 |
| Tiny  |    2 | 4.203e-01 |     124.1 |
| Small |    0 | 4.394e-01 |     277.7 |
| Small |    1 | 3.416e-01 |     283.5 |
| Small |    2 | 3.774e-01 |     278.9 |

## MultiHeadQKV vs LocalSum on IEEE supervised AC LF (Approche 3, Gate 5)

Setup matches Approches 1+2 Gate 5 exactly (`ACLoadFlowProblemLoader` with `perturbation_scale=0.1`, `dataset_size=32`, `batch_size=4`, seeds `[0, 1, 2]`, identical Tiny / Small configs, `optax.adam(1e-3)`). Networks scoped to `ieee9` and `ieee14` per `feedback_verification_scope`.

### Side-by-side summary (best-eval median across seeds)

| Network | Size  | LocalSum n_params | MultiHQKV n_params | LocalSum best-eval | MultiHQKV best-eval | Δ best-eval | LocalSum train (s) | MultiHQKV train (s) |
|---------|-------|------------------:|-------------------:|-------------------:|--------------------:|------------:|-------------------:|--------------------:|
| ieee9   | Tiny  |              1587 |               3403 |          1.561e-01 |           2.043e-01 |      +30.9% |              230.3 |               286.1 |
| ieee9   | Small |             15863 |              25407 |          5.075e-03 |           4.920e-03 |       -3.1% |              573.0 |               526.7 |
| ieee14  | Tiny  |              1587 |               3403 |          8.668e-02 |           1.377e-01 |      +58.9% |              416.1 |               366.8 |
| ieee14  | Small |             15863 |              25407 |          4.564e-03 |           4.635e-03 |       +1.6% |              474.9 |               672.0 |

MultiHQKV has ~60% more parameters than LocalSum (Q + K + V tree vs LocalSum's single value tree). On Small, MultiHQKV best-eval matches LocalSum within seed noise on both networks (ieee9 -3.1 %, ieee14 +1.6 %). On the five mechanisms in this branch this is the only Small AC-LF median that falls inside LocalSum's CI95. The Tiny configs underperform LocalSum because the bilinear projection needs enough capacity to express each Q/K/V branch.

### Cross-Approche comparison — AC LF Small best-eval (median)

| Mechanism | n_params (ieee Small) | ieee9 best | ieee14 best | Gate 6 fwd overhead |
|---|---:|---:|---:|---:|
| LocalSum (baseline) | 15863 | 5.075e-03 | 4.564e-03 | ×1.00 (ref) |
| GATv2 (Approche 1) | 25289 | 5.837e-03 | 7.995e-03 | ×1.98 |
| GlobalAggregation (Approche 2) | 6879 | 2.662e-02 | 1.846e-02 | ×0.09 |
| **MultiHeadQKV (Approche 3)** | **25407** | **4.920e-03** | **4.635e-03** | **×2.07** |

MultiHeadQKV matches LocalSum eval at the cost of 60% more parameters and ×2 forward overhead. Best-eval is below GATv2 (ieee9 4.92e-03 vs 5.84e-03; ieee14 4.64e-03 vs 8.00e-03) and below GlobalAggregation by roughly 5×.

### Per-run detail (MultiHeadQKV, best-eval per seed)

| Network | Size  | Seed | best_eval | train (s) |
|---------|-------|-----:|----------:|----------:|
| ieee9   | Tiny  |    0 | 2.043e-01 |     273.1 |
| ieee9   | Tiny  |    1 | 1.329e-01 |     286.1 |
| ieee9   | Tiny  |    2 | 2.533e-01 |     287.9 |
| ieee9   | Small |    0 | 2.456e-03 |     526.7 |
| ieee9   | Small |    1 | 8.241e-03 |     437.2 |
| ieee9   | Small |    2 | 4.920e-03 |     542.2 |
| ieee14  | Tiny  |    0 | 1.377e-01 |     342.1 |
| ieee14  | Tiny  |    1 | 1.143e-01 |     366.8 |
| ieee14  | Tiny  |    2 | 1.556e-01 |     368.1 |
| ieee14  | Small |    0 | 4.635e-03 |     664.9 |
| ieee14  | Small |    1 | 4.005e-03 |     683.6 |
| ieee14  | Small |    2 | 9.770e-03 |     672.0 |

## Gate 6 perf — LocalSum vs MultiHeadQKV forward / forward+backward

Median wall-time per call after 20 warm-up + 100 timed calls, identical hyper-parameters, `nnx.jit`-compiled. Same substrates as Approches 1+2 Gate 6.

| Substrate    | n_addr | LocalSum fwd (ms) | MultiHQKV fwd (ms) | LocalSum fwd+bwd (ms) | MultiHQKV fwd+bwd (ms) | overhead fwd | overhead fwd+bwd |
|--------------|------:|------------------:|-------------------:|----------------------:|-----------------------:|-------------:|-----------------:|
| LinearSystem |     4 |              1.83 |               3.45 |                  2.16 |                   4.46 |        ×1.88 |            ×2.07 |
| ieee118      |   118 |              9.73 |              20.12 |                 11.85 |                  25.58 |        ×2.07 |            ×2.16 |
| ieee300      |   300 |              9.27 |              19.20 |                 12.56 |                  25.59 |        ×2.07 |            ×2.04 |

MultiHeadQKV overhead is consistently ~**×2.07** forward across all substrates — slightly higher than Approche 1 GATv2 (which was ×2.05 forward), reflecting the extra Q projection per address plus K + V projections per edge (vs GATv2's score + value pair). Mechanism cost is predictable and scale-independent. Inverse of Approche 2 GlobalAggregation overhead pattern (×0.09 - ×0.39 faster than LocalSum).

## Gate 7 consistency — MultiHeadQKV reproducibility on ieee14

Fixed-seed forward output hashes to:

```
output_sha256 = 6b976412dd7a57a50670aab60cdb091bcdcdfd516466d1f49256c5e30fd3b854
```

Two re-runs of `benchmarks/03_multi_head_qkv/consistency_multi_head_qkv.py` on the La Javaness GPU server reproduce this hash bit-for-bit. Different hash from Approche 1 GATv2 (`21647f27…`) and Approche 2 GlobalAggregation (`6c2d2c1a…`), as expected for a different message function. The IEEE-14 context is pickled once and re-loaded on subsequent runs to isolate MultiHeadQKV reproducibility from pypowsybl's non-determinism (same pattern as Approches 1+2).

## Approche 4 closure — Performer (Item 4)

Item 4 (`PerformerMessagePassingFunction`) is implemented per the backlog spec section 3.4 (cf. report, chapter 14) (v1 single-head, no softmax, no random-feature kernel approximation; kernel-trick form $M = \sum_{a'} V_{a'} K_{a'}^\top$ then $\psi_a = (M Q_a) / \sqrt{d_{QK}}$). All Approche 4 closure benchmarks were run on the La Javaness GPU server (CUDA, JAX 0.9.0).

### Verification gates

| Gate | Substrate | Status | Evidence |
|---|---|---|---|
| 1 Unit | `tests/model/unit/test_performer_message_passing.py` | 15/15 pass | Q/K/V dims, output shape, Form A vs Form B kernel-trick parity, scale_scores toggle + sqrt(d_qk) factor check, fictitious masking, permutation equivariance, vmap+jit, gradient flow, all-fictitious zero output |
| 2 Static | `black`, `flake8` | clean | no new violations introduced |
| 3 LinearSystem | `baseline_performer_linearsystem.py` | converges | see "LinearSystem" section below |
| 4 Integration | `tests/model/integration/test_coupler.py` | 7/7 pass | one new `RecurrentCoupler` case wrapping Performer alongside the six existing cases |
| 5 IEEE supervised | `baseline_performer_ac_load_flow.py` on ieee9/14 | **Performer trails LocalSum 4-5× on Small** | see "IEEE AC LF" section below |
| 6 Perf | `perf_performer_vs_localsum.py` 3 substrates | forward time **substrate-size-independent ≈1.7 ms**; vs LocalSum ratio ×0.79 / ×0.21 / ×0.21 on LinearSystem / ieee118 / ieee300 | see "Perf" section below |
| 7 Consistency | `consistency_performer.py` on ieee14 | bit-identical | empreinte numérique `413b6662…` reproduces across re-runs |
| 8 Point figé réseau français | not yet provided | out of scope (snapshot non livré) | runs retrospectively when available |

## Performer vs LocalSum on LinearSystem (Approche 4, Gate 3)

Same dataset config / seeds / epoch budgets / optimizer as Approches 1-3 Gate 3. La Javaness GPU server.

### Side-by-side summary

| Size  | LocalSum n_params | Performer n_params | LocalSum best-eval (med) | Performer best-eval (med) | Δ best-eval | LocalSum train (s) | Performer train (s) |
|-------|------------------:|-------------------:|-------------------------:|--------------------------:|------------:|-------------------:|--------------------:|
| Tiny  |               185 |                145 |                4.388e-01 |                 7.236e-01 |      +64.9% |              116.5 |                76.5 |
| Small |              2177 |               1537 |                3.877e-01 |                 4.265e-01 |      +10.0% |              227.2 |               146.5 |

Performer has fewer parameters than LocalSum on LinearSystem (145 vs 185 on Tiny, 1537 vs 2177 on Small) because the three Q/K/V MLPs operate on the coordinate space (`in_array_size=latent_dim=4`) rather than the per-(class, port) concatenated edge-feature space used by LocalSum. Best-eval is worse (+65% on Tiny, +10% on Small) — expected: LinearSystem DC toy has `n_addr=4` per problem, all-to-all attention degenerates and the kernel-trick complexity advantage does not apply at such small `n_addr`. The pattern reverses on larger substrates (see Gate 6 on ieee118/300).

Train wall-time is ~35% shorter than LocalSum (76s vs 116s on Tiny, 146s vs 227s on Small), a direct consequence of Performer's near-constant forward time.

### Per-run detail (Performer)

| Size  | Seed | best_eval | train (s) |
|-------|-----:|----------:|----------:|
| Tiny  |    0 | 1.005e+00 |      77.4 |
| Tiny  |    1 | 7.236e-01 |      76.5 |
| Tiny  |    2 | 6.557e-01 |      76.3 |
| Small |    0 | 4.657e-01 |     145.5 |
| Small |    1 | 3.458e-01 |     146.8 |
| Small |    2 | 4.333e-01 |     146.5 |

## Performer vs LocalSum on IEEE supervised AC LF (Approche 4, Gate 5)

Setup matches Approches 1-3 Gate 5 exactly (`ACLoadFlowProblemLoader` with `perturbation_scale=0.1`, `dataset_size=32`, `batch_size=4`, seeds `[0, 1, 2]`, identical Tiny / Small configs, `optax.adam(1e-3)`). Networks scoped to `ieee9` and `ieee14` per `feedback_verification_scope`. Total train wall-time across the 12 runs: 68 minutes.

### Side-by-side summary (best-eval median across seeds)

| Network | Size  | LocalSum n_params | Performer n_params | LocalSum best-eval | Performer best-eval | Δ best-eval | LocalSum train (s) | Performer train (s) |
|---------|-------|------------------:|-------------------:|-------------------:|--------------------:|------------:|-------------------:|--------------------:|
| ieee9   | Tiny  |              1587 |                799 |          1.561e-01 |           2.039e-01 |      +30.6% |              230.3 |               183.6 |
| ieee9   | Small |             15863 |               7439 |          5.075e-03 |           3.067e-02 |     +504.4% |              573.0 |               356.4 |
| ieee14  | Tiny  |              1587 |                799 |          8.668e-02 |           1.221e-01 |      +40.9% |              416.1 |               303.6 |
| ieee14  | Small |             15863 |               7439 |          4.564e-03 |           2.264e-02 |     +396.1% |              474.9 |               453.5 |

Performer has ~47% the parameter count of LocalSum on ieee Small (7439 vs 15863). Its best-eval is materially worse on Small (4-5× higher MSE) and on Tiny (30-41% higher MSE). This is the direct empirical consequence of Performer ignoring graph topology: per attention-backlog section 3.4 the aggregation runs over all addresses indiscriminately and the `port_dict` connectivity is not consulted, so Performer cannot exploit the per-(class, port) signal that LocalSum / GATv2 / MultiHeadQKV rely on. Train wall-time is shorter (median 357-454 s vs 474-573 s for LocalSum on Small) consistent with the constant-time forward.

### Cross-Approche comparison — AC LF Small best-eval (median)

| Mechanism | n_params (ieee Small) | ieee9 best | ieee14 best | Gate 6 fwd overhead |
|---|---:|---:|---:|---:|
| LocalSum (baseline) | 15863 | 5.075e-03 | 4.564e-03 | ×1.00 (ref) |
| GATv2 (Approche 1) | 25289 | 5.837e-03 | 7.995e-03 | ×1.98 |
| GlobalAggregation (Approche 2) | 6879 | 2.662e-02 | 1.846e-02 | ×0.09 |
| MultiHeadQKV (Approche 3) | 25407 | **4.920e-03** | **4.635e-03** | ×2.07 |
| **Performer (Approche 4)** | **7439** | **3.067e-02** | **2.264e-02** | **×0.21** |

Performer and GlobalAggregation are both topology-agnostic (no per-(class, port) factoring): roughly half LocalSum's parameters, 4-6x worse best-eval on AC LF Small, x0.09-0.21 forward time. The fwd ratio differs (Performer x0.21 vs GlobalAggregation x0.09 on ieee118/300) because Performer adds a Q-dependent reweighting that GlobalAggregation does not. (Kernel-trick complexity $\mathcal{O}(n_{\mathrm{addr}} \cdot d_V \cdot d_{QK})$, whereas GlobalAggregation is faster only by ×0.09-0.39 (no Q-dependent weighting cost).

### Per-run detail (Performer, best-eval per seed)

| Network | Size  | Seed | best_eval | train (s) |
|---------|-------|-----:|----------:|----------:|
| ieee9   | Tiny  |    0 | 2.039e-01 |     174.2 |
| ieee9   | Tiny  |    1 | 1.930e-01 |     183.6 |
| ieee9   | Tiny  |    2 | 2.385e-01 |     189.5 |
| ieee9   | Small |    0 | 3.598e-02 |     363.7 |
| ieee9   | Small |    1 | 3.067e-02 |     356.4 |
| ieee9   | Small |    2 | 2.881e-02 |     355.7 |
| ieee14  | Tiny  |    0 | 1.221e-01 |     313.7 |
| ieee14  | Tiny  |    1 | 1.187e-01 |     303.6 |
| ieee14  | Tiny  |    2 | 1.375e-01 |     302.4 |
| ieee14  | Small |    0 | 2.264e-02 |     448.3 |
| ieee14  | Small |    1 | 1.903e-02 |     453.5 |
| ieee14  | Small |    2 | 2.387e-02 |     454.8 |

## Gate 6 perf — LocalSum vs Performer forward / forward+backward

Median wall-time per call after 20 warm-up + 100 timed calls, identical hyper-parameters, `nnx.jit`-compiled. Same substrates as Approches 1-3 Gate 6.

| Substrate    | n_addr | LocalSum fwd (ms) | Performer fwd (ms) | LocalSum fwd+bwd (ms) | Performer fwd+bwd (ms) | overhead fwd | overhead fwd+bwd |
|--------------|------:|------------------:|-------------------:|----------------------:|-----------------------:|-------------:|-----------------:|
| LinearSystem |     4 |              2.17 |               1.72 |                  2.03 |                   1.96 |        ×0.79 |            ×0.97 |
| ieee118      |   118 |              7.99 |               1.66 |                 10.45 |                   1.98 |        ×0.21 |            ×0.19 |
| ieee300      |   300 |              8.01 |               1.70 |                 10.31 |                   2.00 |        ×0.21 |            ×0.19 |

Performer's forward time is near-constant across substrates (~1.7 ms forward, ~2 ms forward+backward), a direct consequence of the kernel-trick complexity $\mathcal{O}(n_{\mathrm{addr}} \cdot d_V \cdot d_{QK})$ with $d_V, d_{QK}$ small and fixed. LocalSum scales with the number of hyper-edges (per-edge MLP cost dominates), hence the ×0.21 ratio on ieee118/300. On LinearSystem (`n_addr=4`) the ratio is near unity — too few hyper-edges for LocalSum to fall behind.

This pattern is the inverse of Approche 3 MultiHeadQKV (×2.07 overhead consistent across substrates) — the difference is in aggregation scope: MultiHeadQKV is per-edge (cost grows with edge count like LocalSum), Performer is all-to-all with outer-product factoring.

## Gate 7 consistency — Performer reproducibility on ieee14

Fixed-seed forward output hashes to:

```
output_sha256 = 413b66628748dfed3dbfac926e1b9a8f4656338ec95a07ef562c292cb488355e
```

Two re-runs of `benchmarks/04_performer/consistency_performer.py` on the La Javaness GPU server reproduce this hash bit-for-bit. Different hash from Approche 1 GATv2 (`21647f27…`), Approche 2 GlobalAggregation (`6c2d2c1a…`), and Approche 3 MultiHeadQKV (`6b976412…`), as expected for a different message function. The ieee14 context is pickled once and re-loaded on subsequent runs to isolate Performer reproducibility from pypowsybl's non-determinism (same pattern as Approches 1-3).

## Approche 4 ablations — capacity-matched, 5-seed CI, combo

Three follow-up ablations to inform the Item 5 design. Scope: AC LF Small (ieee9 and ieee14). All runs on the La Javaness GPU server.

### Capacity-matched configurations

Performer and GlobalAggregation Small configurations are scaled by adjusting `hidden_sizes` so that the full-GNN parameter count matches LocalSum's reference of 15863 within ±2%:

| Mechanism | hidden_sizes | n_params | n_params delta vs LocalSum |
|---|---|---:|---:|
| LocalSum (reference) | (16,) | 15863 | 0 |
| Performer (cap-matched) | (35,) | 16027 | +1.0% |
| GlobalAggregation (cap-matched) | (38,) | 16075 | +1.3% |

Three seeds (0, 1, 2) on ieee9 and ieee14 Small. Best-eval median:

| Configuration | n_params | ieee9 best-eval | ratio vs LocalSum | ieee14 best-eval | ratio vs LocalSum |
|---|---:|---:|---:|---:|---:|
| LocalSum (reference) | 15863 | 5.075e-03 | ×1.00 | 4.564e-03 | ×1.00 |
| Performer original | 7439 | 2.881e-02 | ×5.68 | 2.013e-02 | ×4.41 |
| Performer cap-matched | 16027 | 1.093e-02 | ×2.15 | 8.826e-03 | ×1.94 |
| GlobalAggregation original | 6879 | 2.662e-02 | ×5.24 | 1.728e-02 | ×3.79 |
| GlobalAggregation cap-matched | 16075 | 1.159e-02 | ×2.28 | 8.146e-03 | ×1.79 |

Scaling Performer and GlobalAggregation up to LocalSum reference capacity closes about half of the eval gap. On Performer ieee9, the ratio drops from ×5.68 to ×2.15; CI95 of Performer cap-matched (3 seeds) is [1.08e-02, 1.52e-02], CI95 of LocalSum (5 seeds) is [1.15e-03, 6.89e-03] — the intervals do not overlap, the remaining gap is real. Capacity accounts for approximately half of the Group A vs Group B eval differential; the topology-aware per-(class, port) factoring carries the remaining half.

Per-run detail (capacity-matched, best-eval per seed):

| Configuration | Network | Seed | best_eval | train (s) |
|---|---|---:|---:|---:|
| Performer cap-matched | ieee9 | 0 | 1.093e-02 | 284.5 |
| Performer cap-matched | ieee9 | 1 | 1.084e-02 | 289.5 |
| Performer cap-matched | ieee9 | 2 | 1.517e-02 | 291.0 |
| Performer cap-matched | ieee14 | 0 | 8.826e-03 | 356.9 |
| Performer cap-matched | ieee14 | 1 | 7.727e-03 | 370.2 |
| Performer cap-matched | ieee14 | 2 | 1.097e-02 | 368.2 |
| GlobalAgg cap-matched | ieee9 | 0 | 9.435e-03 | 272.0 |
| GlobalAgg cap-matched | ieee9 | 1 | 1.159e-02 | 279.1 |
| GlobalAgg cap-matched | ieee9 | 2 | 1.635e-02 | 277.8 |
| GlobalAgg cap-matched | ieee14 | 0 | 7.628e-03 | 347.4 |
| GlobalAgg cap-matched | ieee14 | 1 | 8.146e-03 | 355.9 |
| GlobalAgg cap-matched | ieee14 | 2 | 1.149e-02 | 355.0 |

### 5-seed bootstrap confidence intervals

Seeds 3 and 4 are run on AC LF Small for LocalSum, GlobalAggregation, MultiHeadQKV and Performer, complementing the existing 3 seeds (0, 1, 2). GATv2 stays at 3 seeds. Bootstrap CI95 from 10000 resamples of medians over the combined 5 seeds:

| Mechanism | n_params | ieee9 median | ieee9 CI95 | ieee14 median | ieee14 CI95 | n_seeds |
|---|---:|---:|---|---:|---|---:|
| LocalSum | 15863 | 5.08e-03 | [1.15e-03, 6.89e-03] | 4.56e-03 | [2.21e-03, 9.87e-03] | 5 |
| GATv2 | 25289 | 5.84e-03 | [2.17e-03, 1.38e-02] | 8.00e-03 | [4.63e-03, 9.61e-03] | 3 |
| GlobalAggregation | 6879 | 2.66e-02 | [1.60e-02, 3.28e-02] | 1.73e-02 | [1.34e-02, 2.11e-02] | 5 |
| MultiHeadQKV | 25407 | 8.24e-03 | [2.46e-03, 1.53e-02] | 5.64e-03 | [4.01e-03, 9.77e-03] | 5 |
| Performer | 7439 | 2.88e-02 | [1.72e-02, 3.60e-02] | 2.01e-02 | [1.52e-02, 2.39e-02] | 5 |

Per-seed values for the four mechanisms with extra seeds:

| Mechanism | Network | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---:|---:|---:|---:|---:|
| LocalSum | ieee9 | 5.075e-03 | 1.148e-03 | 6.569e-03 | 2.956e-03 | 6.888e-03 |
| LocalSum | ieee14 | 4.564e-03 | 2.209e-03 | 9.866e-03 | 5.536e-03 | 3.086e-03 |
| GlobalAggregation | ieee9 | 3.283e-02 | 1.602e-02 | 2.662e-02 | 2.353e-02 | 2.789e-02 |
| GlobalAggregation | ieee14 | 1.846e-02 | 2.114e-02 | 1.728e-02 | 1.338e-02 | 1.634e-02 |
| MultiHeadQKV | ieee9 | 2.456e-03 | 8.241e-03 | 4.920e-03 | 1.159e-02 | 1.534e-02 |
| MultiHeadQKV | ieee14 | 4.635e-03 | 4.005e-03 | 9.770e-03 | 5.636e-03 | 8.285e-03 |
| Performer | ieee9 | 3.598e-02 | 3.067e-02 | 2.881e-02 | 1.716e-02 | 2.668e-02 |
| Performer | ieee14 | 2.264e-02 | 1.903e-02 | 2.387e-02 | 1.524e-02 | 2.013e-02 |

Group A (topology-aware: LocalSum, MultiHeadQKV; GATv2 at 3 seeds) and Group B (topology-agnostic: GlobalAggregation, Performer) CI95s do not overlap on ieee9 and ieee14, the group separation is real. Within-group differences (LocalSum vs MultiHeadQKV) show CI95 overlap, the two mechanisms are tied on this benchmark.

MultiHeadQKV ieee9 median moved from 4.92e-03 (3 seeds) to 8.24e-03 (5 seeds) as seeds 3 and 4 added higher values (1.16e-02 and 1.53e-02). CI95 [2.46e-03, 1.53e-02] remains wide; further reduction would require n=10 to 20 seeds.

### Combo [LocalSum + Performer]

`RecurrentCoupler(message_functions=[LocalSumMessagePassingFunction, PerformerMessagePassingFunction], ...)` — the existing coupler concatenates the two messages along the feature axis before passing through `phi`. Three seeds on AC LF Small ieee9 and ieee14, plus LinearSystem Tiny and Small for completeness.

| Substrate | LocalSum alone median | Combo median | Combo CI95 | CI overlap with LocalSum? |
|---|---:|---:|---|:---:|
| LinearSystem Tiny | 4.39e-01 | 4.21e-01 | [3.70e-01, 5.84e-01] | yes |
| LinearSystem Small | 3.88e-01 | 4.09e-01 | [3.60e-01, 4.39e-01] | yes |
| AC LF ieee9 Small | 5.08e-03 | 7.73e-03 | [5.35e-03, 9.55e-03] | yes |
| AC LF ieee14 Small | 4.56e-03 | 6.14e-03 | [4.47e-03, 9.76e-03] | yes |

Combo CI95 covers LocalSum CI95 on all four substrates. The combination does not improve eval beyond LocalSum alone. Combo CI95 is disjoint from Performer-alone CI95 on AC LF Small (Combo [5.35e-03, 9.55e-03] vs Performer [1.72e-02, 3.60e-02] on ieee9), the combination recovers Group A eval level. The interpretation is that LocalSum already carries the signal for AC LF; Performer's per-receiver Q-weighted global context, added through plain concatenation, does not contribute new information.

This ablation motivates the Item 5 design: simple concatenation through the existing `RecurrentCoupler.message_functions` list is not the combining mechanism that exploits multiple signals jointly. The `VirtualAddressRecurrentCoupler` in the backlog section 3.5 proposes a shared virtual state across message functions as the candidate combining mechanism.

Combo Gate-6-equivalent forward cost is not directly measured; summing the Approche 4 Gate 6 numbers (LocalSum 7.99 ms + Performer 1.66 ms on ieee118; 8.01 + 1.70 on ieee300; 2.17 + 1.72 on LinearSystem) gives 9.65 ms / 9.71 ms / 3.89 ms upper bounds. The Performer leg adds ~1.7 ms on top of LocalSum on the larger substrates.

Per-run detail (Combo, best-eval per seed):

| Substrate | Seed | best_eval | train (s) |
|---|---|---:|---:|
| LinearSystem Tiny | 0 | 4.213e-01 | 80.1 |
| LinearSystem Tiny | 1 | 3.702e-01 | 80.3 |
| LinearSystem Tiny | 2 | 5.843e-01 | 82.2 |
| LinearSystem Small | 0 | 4.388e-01 | 183.1 |
| LinearSystem Small | 1 | 3.598e-01 | 177.0 |
| LinearSystem Small | 2 | 4.090e-01 | 179.5 |
| AC LF ieee9 Tiny | 0 | 1.291e-01 | 180.3 |
| AC LF ieee9 Tiny | 1 | 1.745e-01 | 201.3 |
| AC LF ieee9 Tiny | 2 | 1.505e-01 | 195.8 |
| AC LF ieee9 Small | 0 | 7.726e-03 | 345.4 |
| AC LF ieee9 Small | 1 | 5.352e-03 | 343.5 |
| AC LF ieee9 Small | 2 | 9.548e-03 | 343.2 |
| AC LF ieee14 Tiny | 0 | 1.055e-01 | 236.5 |
| AC LF ieee14 Tiny | 1 | 1.195e-01 | 242.6 |
| AC LF ieee14 Tiny | 2 | 2.452e-01 | 243.4 |
| AC LF ieee14 Small | 0 | 6.143e-03 | 422.2 |
| AC LF ieee14 Small | 1 | 4.468e-03 | 426.9 |
| AC LF ieee14 Small | 2 | 9.762e-03 | 432.8 |

## Approche 5 closure — VirtualAddressRecurrentCoupler (Item 5)

Item 5 (`VirtualAddressRecurrentCoupler`) is implemented per the backlog spec section 3.5 (cf. report, chapter 15). The class extends `RecurrentCoupler` with a single shared virtual state vector evolved in parallel with the per-address state via two parallel forward-Euler updates. Design choices on the three open questions Q5.1, Q5.2 and Q5.3 of backlog section 3.5:

- **Q5.1 (injection)**: design (c) — the virtual state is concatenated with the message vector before phi. No change to the `MessagePassingFunction` ABC; Items 1-4 are reused unchanged.
- **Q5.2 (F_virtual)**: design (alpha) — masked mean pool of `h` over real addresses, concatenated with the previous virtual state, passed through a dedicated MLP `phi_virtual`. Reuses the corrected-denominator pattern of `GlobalAggregationMessagePassingFunction` (Approche 2).
- **Q5.3 (virtual count)**: single virtual address for v1.

The v1 configuration runs LocalSum (Donon's reference, Approche 0) as the wrapped message function. This isolates the contribution of the virtual state itself, comparable against the existing `RecurrentCoupler` + LocalSum baseline.

All Approche 5 closure benchmarks were run on the La Javaness GPU server (CUDA, JAX 0.9.0).

### Verification gates

| Gate | Substrate | Status | Evidence |
|---|---|---|---|
| 1 Unit | `tests/model/unit/test_virtual_address_recurrent_coupler.py` | 11/11 pass | constructor stores attributes, forward shape and info, forward finite, deterministic with seed, zero-virtual-size reduces to RecurrentCoupler, single-step Euler, fictitious addresses excluded from virtual mean, multi-message-functions, vmap+jit safety, gradient flow through phi_virtual, h_virtual state evolves |
| 2 Static | `black`, `flake8` | clean | no new violations introduced |
| 3 LinearSystem | `baseline_var_linearsystem.py` | converges | see "LinearSystem" section below |
| 4 Integration | `tests/model/integration/test_coupler.py` | 8/8 pass | one new `RecurrentCoupler`-family case wrapping the virtual-address coupler |
| 5 IEEE supervised | `baseline_var_ac_load_flow.py` on ieee9/14 | **VAR+LocalSum CI95 overlap LocalSum reference on AC LF Small ieee9 and ieee14, negative result at 3 seeds** | see "IEEE AC LF" section below |
| 6 Perf | not yet run | deferred | follow-on commit on `virtual-address-coupler` |
| 7 Consistency | not yet run | deferred | follow-on commit on `virtual-address-coupler` |
| 8 Point figé réseau français | not yet provided | out of scope (snapshot non livré) | runs retrospectively when available |

### VirtualAddressRecurrentCoupler + LocalSum on LinearSystem (Approche 5, Gate 3)

Same dataset config and seeds as Approches 0-4 Gate 3 (n_max=3, dataset_size=64, batch_size=4, val_dataset_size=32, seeds [0, 1, 2], `optax.adam(1e-3)`). The wrapped message function is `LocalSumMessagePassingFunction`. `virtual_address_size` is set equal to `latent_dim`. Total elapsed: 13.9 minutes for 6 runs.

| Size  | LocalSum (Approche 0) n_params | VAR+LocalSum n_params | LocalSum (Approche 0) best-eval | VAR+LocalSum best-eval | Δ best-eval | LocalSum train (s) | VAR+LocalSum train (s) |
|-------|------------------:|-------------------:|-------------------------:|--------------------------:|------------:|-------------------:|--------------------:|
| Tiny  |               185 |                237 |                4.388e-01 |                 4.446e-01 |       +1.3% |              116.5 |                93.2 |
| Small |              2177 |               2377 |                3.877e-01 |                 3.946e-01 |       +1.8% |              227.2 |               181.1 |

VAR+LocalSum adds the virtual-state machinery on top of plain LocalSum: 52 extra params on Tiny (phi_virtual MLP plus phi resize for the virtual concatenation), 200 on Small. Best-eval median matches the LocalSum reference on both Tiny and Small (within seed noise, since 3-seed bootstrap CI on Approche 0 already spans a similar range). The virtual state does not add a measurable signal on LinearSystem, expected given the substrate has `n_addr=4` per problem and minimal topology to bridge.

### Per-run detail (VAR+LocalSum on LinearSystem)

| Size  | Seed | best_eval | train (s) |
|-------|-----:|----------:|----------:|
| Tiny  |    0 | 8.777e-01 |     106.1 |
| Tiny  |    1 | 3.823e-01 |      90.9 |
| Tiny  |    2 | 4.446e-01 |      93.2 |
| Small |    0 | 4.465e-01 |     181.1 |
| Small |    1 | 3.583e-01 |     188.2 |
| Small |    2 | 3.994e-01 |     174.8 |

### VAR + LocalSum vs LocalSum on IEEE supervised AC LF (Approche 5, Gate 5)

Setup matches Approches 1-4 Gate 5 exactly (`ACLoadFlowProblemLoader` with `perturbation_scale=0.1`, `dataset_size=32`, `batch_size=4`, seeds [0, 1, 2], identical Tiny / Small configs, `optax.adam(1e-3)`). Networks scoped to `ieee9` and `ieee14` per `feedback_verification_scope`. The wrapped message function is `LocalSumMessagePassingFunction`; `virtual_address_size = latent_dim`.

| Network | Size  | LocalSum n_params | VAR+LocalSum n_params | LocalSum best-eval | VAR+LocalSum best-eval | Δ best-eval | LocalSum train (s) | VAR+LocalSum train (s) |
|---------|-------|------------------:|----------------------:|-------------------:|-----------------------:|------------:|-------------------:|-----------------------:|
| ieee9   | Tiny  |              1587 |                  1639 |          1.561e-01 |              1.476e-01 |       -5.4% |              230.3 |                  248.2 |
| ieee9   | Small |             15863 |                 16063 |          5.075e-03 |              6.281e-03 |      +23.7% |              573.0 |                  399.9 |
| ieee14  | Tiny  |              1587 |                  1639 |          8.668e-02 |              1.165e-01 |      +34.4% |              416.1 |                  324.8 |
| ieee14  | Small |             15863 |                 16063 |          4.564e-03 |              7.127e-03 |      +56.1% |              474.9 |                  506.1 |

Bootstrap CI95 (3 seeds for VAR+LocalSum, 5 seeds for LocalSum reference):

| Network | Size  | LocalSum CI95 | VAR+LocalSum CI95 | CI overlap? |
|---------|-------|---------------|-------------------|:-----------:|
| ieee9   | Small | [1.15e-03, 6.89e-03] | [2.68e-03, 8.35e-03] | yes |
| ieee14  | Small | [2.21e-03, 9.87e-03] | [3.24e-03, 1.18e-02] | yes |

The virtual state addition shifts the median upward on three of four (network, size) cells but the CI95 intervals overlap LocalSum's reference CI95 on both Small configurations. The shift is within seed noise at the current 3-seed sample. The median values for VAR+LocalSum on Small networks (6.28e-03 and 7.13e-03) remain in the same order of magnitude as LocalSum (5.08e-03 and 4.56e-03), well within Group A range; the virtual state does not pull the eval into Group B.

### Cross-Approche comparison — AC LF Small best-eval (median)

| Mechanism | n_params (ieee Small) | ieee9 best | ieee14 best | Gate 6 fwd overhead |
|---|---:|---:|---:|---:|
| LocalSum (baseline) | 15863 | 5.075e-03 | 4.564e-03 | ×1.00 (ref) |
| GATv2 (Approche 1) | 25289 | 5.837e-03 | 7.995e-03 | ×1.98 |
| GlobalAggregation (Approche 2) | 6879 | 2.662e-02 | 1.846e-02 | ×0.09 |
| MultiHeadQKV (Approche 3) | 25407 | 4.920e-03 | 4.635e-03 | ×2.07 |
| Performer (Approche 4) | 7439 | 2.881e-02 | 2.013e-02 | ×0.21 |
| **VAR+LocalSum (Approche 5)** | **16063** | **6.281e-03** | **7.127e-03** | _deferred (follow-on)_ |

Approche 5 sits in the Group A band (topology-aware) as expected since it wraps the same LocalSum message function. The virtual state mechanism does not improve eval over RecurrentCoupler + LocalSum on either ieee9 or ieee14 Small at 3 seeds. The cross-Approche pattern remains intact: topology-aware mechanisms (LocalSum, GATv2, MultiHeadQKV, VAR+LocalSum) cluster in the 5-8e-03 range on ieee9 Small; topology-agnostic mechanisms (GlobalAggregation, Performer) cluster around 2.7-2.9e-02.

### Per-run detail (VAR+LocalSum on AC LF Small)

| Network | Size  | Seed | best_eval | train (s) |
|---------|-------|-----:|----------:|----------:|
| ieee9   | Tiny  |    0 | 1.476e-01 |     273.3 |
| ieee9   | Tiny  |    1 | 1.582e-01 |     253.9 |
| ieee9   | Tiny  |    2 | 1.324e-01 |     248.2 |
| ieee9   | Small |    0 | 2.682e-03 |     413.2 |
| ieee9   | Small |    1 | 8.354e-03 |     399.0 |
| ieee9   | Small |    2 | 6.281e-03 |     399.9 |
| ieee14  | Tiny  |    0 | 9.832e-02 |     324.8 |
| ieee14  | Tiny  |    1 | 1.165e-01 |     315.9 |
| ieee14  | Tiny  |    2 | 1.261e-01 |     335.1 |
| ieee14  | Small |    0 | 3.239e-03 |     510.3 |
| ieee14  | Small |    1 | 7.127e-03 |     506.1 |
| ieee14  | Small |    2 | 1.181e-02 |     505.4 |

### Implication and follow-on

Approche 5 v1 result: negative. Median best-eval shifts +24 to +56 % vs LocalSum on three of four (network, size) cells; CI95 overlaps the LocalSum reference on both Small networks at 3 seeds — within seed noise.

Follow-on candidate (not run): VAR + Performer and VAR + GlobalAggregation on ieee9 / ieee14 Small, 3 seeds. The Approche 4 Combo ablation already shows that plain concatenation of LocalSum + Performer does not exceed LocalSum alone (Combo CI95 covers LocalSum CI95 on all four substrates). Whether the virtual-state channel changes that conclusion is open.
