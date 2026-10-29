# tplan (skeleton light per lateness slice tau = t - depth / c), 0.6.3

11018 training / 1377 validation skeletons.  3 points per stroke, as showerlm.gfeat / zplan.

| lateness slice (ns) | mean photons | mean fraction | fraction p5 | p50 | p95 | std / mean of fraction | val mean fraction |
|---|---|---|---|---|---|---|---|
| 0-0.25 | 1.11e+08 | 0.8228 | 0.7317 | 0.8239 | 0.9071 | 0.07 | 0.8248 |
| 0.25-0.5 | 5.1e+06 | 0.0395 | 0.0206 | 0.0392 | 0.0598 | 0.30 | 0.0392 |
| 0.5-1 | 5.28e+06 | 0.0409 | 0.0214 | 0.0405 | 0.0620 | 0.30 | 0.0404 |
| 1-2 | 4.99e+06 | 0.0386 | 0.0200 | 0.0383 | 0.0589 | 0.31 | 0.0381 |
| 2-4 | 3.97e+06 | 0.0307 | 0.0157 | 0.0304 | 0.0470 | 0.31 | 0.0304 |
| 4-8 | 2.42e+06 | 0.0187 | 0.0092 | 0.0184 | 0.0293 | 0.33 | 0.0185 |
| >8 | 1.14e+06 | 0.0088 | 0.0039 | 0.0086 | 0.0146 | 0.37 | 0.0087 |

Skeleton photons per shower: mean 1.338e+08, std 1.58e+07.
