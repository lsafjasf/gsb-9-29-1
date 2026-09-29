# Morphology benchmark

- Generated: 2026-09-30T05:08:01
- Python: 3.12.3 (Linux-6.18.33.1-microsoft-standard-WSL2-x86_64-with-glibc2.39)
- Image: 64x64
- Structuring element: 31x31 flat rectangle (961 members)
- Timing: minimum of 2 brute runs and 5 separable runs
- All brute-force and separable results are asserted exactly equal

| Dataset | Operation | Brute (s) | Separable (s) | Speedup |
| --- | --- | ---: | ---: | ---: |
| binary | erosion | 0.168661 | 0.002307 | 73.1x |
| binary | dilation | 0.160711 | 0.002290 | 70.2x |
| binary | opening | 0.327879 | 0.004538 | 72.2x |
| binary | closing | 0.352320 | 0.004789 | 73.6x |
| grayscale | erosion | 0.170102 | 0.002479 | 68.6x |
| grayscale | dilation | 0.174471 | 0.002523 | 69.1x |
| grayscale | opening | 0.342714 | 0.005270 | 65.0x |
| grayscale | closing | 0.367789 | 0.005097 | 72.2x |

| Dataset | Four-operation total brute (s) | Total separable (s) | Total speedup |
| --- | ---: | ---: | ---: |
| binary | 1.009571 | 0.013925 | 72.5x |
| grayscale | 1.055075 | 0.015369 | 68.6x |
