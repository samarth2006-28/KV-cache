# McNemar, 3B, 20% memory, n=40 (paired by problem)

## tiered vs lexico
| budget | metric | n | tiered % | lexico % | only tiered | only lexico | McNemar p |
|---|---|---|---|---|---|---|---|
| 20% | flexible | 40 | 35.0 | 32.5 | 9 | 8 | 1 |
| 20% | strict | 40 | 0.0 | 20.0 | 0 | 8 | 0.00781 |

## tiered vs h2o
| budget | metric | n | tiered % | h2o % | only tiered | only h2o | McNemar p |
|---|---|---|---|---|---|---|---|
| 20% | flexible | 40 | 35.0 | 10.0 | 10 | 0 | 0.00195 |
| 20% | strict | 40 | 0.0 | 5.0 | 0 | 2 | 0.5 |

## lexico vs h2o
| budget | metric | n | lexico % | h2o % | only lexico | only h2o | McNemar p |
|---|---|---|---|---|---|---|---|
| 20% | flexible | 40 | 32.5 | 10.0 | 12 | 3 | 0.0352 |
| 20% | strict | 40 | 20.0 | 5.0 | 8 | 2 | 0.109 |
