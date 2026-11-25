# Regularization: Selection, Shrinkage and Valid Intervals

## Objective
I tested when LASSO results can and cannot be read as causal effects, using simulated data where I knew every true coefficient.

## Methodology
- Simulated a treatment effect of 1.0 with 100 candidate controls, five of them real confounders.
- Ran the common procedure: LASSO picks the controls, then OLS on the survivors, and checked how often its 95% interval covered the truth over 200 samples.
- Wrote double selection (Belloni, Chernozhukov and Hansen 2014): one LASSO of the outcome on the controls, one of the treatment on the controls, then OLS on the union.
- Compared LASSO coefficients with the known true ones, and refit OLS on the kept variables (post-LASSO).
- Wrote my own 5-fold cross-validation loop over the penalty, with the scaler fitted inside every fold, and checked it against GridSearchCV.
- Saved the functions in `src/sparse_tools.py`.

## Key Findings
- The LASSO-then-OLS interval covered the truth in only 82.5% of 200 samples, not 95%, because its standard error ignores that the controls were chosen on the same data.
- Double selection raised coverage to 88.5%, keeping 26.5 controls on average.
- LASSO shrank the true coefficients by 4.6% on average, most of all the small ones, so a LASSO coefficient should not be quoted as an effect. Post-LASSO OLS brought the sizes back close to the truth.
- My cross-validation loop matched GridSearchCV exactly: lambda 0.0501, CV MSE 1.0173.
