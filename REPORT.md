# 平滑前后对比数据

```
Smoothing trade-off report
(length / max curvature / min clearance / colliding segment-obstacle pairs)

[narrow_corridor]
  variant    length maxcurv minclear   hits  status
  raw        12.000  0.7071  0.4000      0 safe
  legacy     10.638  0.5562 -0.3722      8 COLLIDES
  fixed      11.276  0.2336  0.0527      0 safe

[sharp_turns]
  variant    length maxcurv minclear   hits  status
  raw        16.000  0.4472  0.2000      0 safe
  legacy     12.869  0.1631 -0.4863      8 COLLIDES
  fixed      12.885  0.0629  0.0520      0 safe

[obstacle_hugging]
  variant    length maxcurv minclear   hits  status
  raw        10.239  0.0839  0.0743      0 safe
  legacy     10.036  0.0336 -0.5784      2 COLLIDES
  fixed      10.228  0.0822  0.0505      0 safe

[single_point]
  variant    length maxcurv minclear   hits  status
  raw         0.000  0.0000    inf      0 safe
  legacy      0.000  0.0000    inf      0 safe
  fixed       0.000  0.0000    inf      0 safe

[straight_line]
  variant    length maxcurv minclear   hits  status
  raw        10.000  0.0000  4.0000      0 safe
  legacy     10.000  0.0000  4.0000      0 safe
  fixed      10.000  0.0000  4.0000      0 safe

[unsmoothable]
  variant    length maxcurv minclear   hits  status
  raw         8.944  0.4000  0.0200      0 safe
  legacy      8.128  0.1587 -0.3451     34 COLLIDES
  fixed       8.944  0.4000  0.0200      0 safe
```
