# Coverage simulation results

Nominal level: 95%.  R = 1000 Monte Carlo replications, B = 999 bootstrap resamples, statistic = sample mean.
Coverage CI = Wilson score interval for the empirical coverage.
Reproduce with `python3 coverage_simulation.py` (fixed seeds; results are deterministic).

| distribution | n | method | coverage | Wilson 95% CI | mean width |
|---|---|---|---|---|---|
| normal | 15 | normal | 0.916 | [0.897, 0.932] | 0.952 |
| normal | 15 | percentile | 0.912 | [0.893, 0.928] | 0.946 |
| normal | 15 | bc | 0.910 | [0.891, 0.926] | 0.948 |
| normal | 15 | bca | 0.911 | [0.892, 0.927] | 0.954 |
| normal | 50 | normal | 0.946 | [0.930, 0.958] | 0.546 |
| normal | 50 | percentile | 0.946 | [0.930, 0.958] | 0.543 |
| normal | 50 | bc | 0.943 | [0.927, 0.956] | 0.544 |
| normal | 50 | bca | 0.945 | [0.929, 0.958] | 0.544 |
| exponential | 15 | normal | 0.887 | [0.866, 0.905] | 0.929 |
| exponential | 15 | percentile | 0.888 | [0.867, 0.906] | 0.917 |
| exponential | 15 | bc | 0.897 | [0.877, 0.914] | 0.932 |
| exponential | 15 | bca | 0.903 | [0.883, 0.920] | 0.979 |
| exponential | 50 | normal | 0.918 | [0.899, 0.933] | 0.535 |
| exponential | 50 | percentile | 0.920 | [0.902, 0.935] | 0.532 |
| exponential | 50 | bc | 0.917 | [0.898, 0.933] | 0.536 |
| exponential | 50 | bca | 0.919 | [0.900, 0.934] | 0.549 |
| bimodal | 15 | normal | 0.910 | [0.891, 0.926] | 2.163 |
| bimodal | 15 | percentile | 0.925 | [0.907, 0.940] | 2.146 |
| bimodal | 15 | bc | 0.928 | [0.910, 0.942] | 2.150 |
| bimodal | 15 | bca | 0.942 | [0.926, 0.955] | 2.158 |
| bimodal | 50 | normal | 0.947 | [0.931, 0.959] | 1.224 |
| bimodal | 50 | percentile | 0.947 | [0.931, 0.959] | 1.216 |
| bimodal | 50 | bc | 0.946 | [0.930, 0.958] | 1.217 |
| bimodal | 50 | bca | 0.949 | [0.934, 0.961] | 1.217 |
| outliers | 15 | normal | 0.964 | [0.951, 0.974] | 2.769 |
| outliers | 15 | percentile | 0.901 | [0.881, 0.918] | 2.678 |
| outliers | 15 | bc | 0.881 | [0.859, 0.900] | 2.780 |
| outliers | 15 | bca | 0.814 | [0.789, 0.837] | 3.126 |
| outliers | 50 | normal | 0.959 | [0.945, 0.970] | 1.704 |
| outliers | 50 | percentile | 0.911 | [0.892, 0.927] | 1.694 |
| outliers | 50 | bc | 0.885 | [0.864, 0.903] | 1.719 |
| outliers | 50 | bca | 0.831 | [0.807, 0.853] | 1.813 |
