# Multi-Omics Survival Prediction in ccRCC

Does MOFA-based multi-omics integration improve survival prediction in clear cell renal cell carcinoma (TCGA-KIRC), compared to single-omics or naive concatenation?

## Data

TCGA-KIRC (Xena Browser): RNA-seq, DNA methylation (450K), survival endpoints.
Reference cohort: 318 samples, 104 events (32.7%), 5 fixed stratified CV folds.

## Files (run in this order)

| # | File | Purpose |
|---|---|---|
| 1 | `00_preprocessing_kirc_multiomics_v2.ipynb` | Load, filter, impute RNA-seq + methylation |
| 2 | `00_build_reference_cohort.ipynb` | Fixed sample set + CV folds, reused everywhere below |
| 3 | `common_utils.py` | Shared modeling functions (imported by 01/02/03) |
| 4 | `01_model_expression.ipynb` | Single-omics baseline (5,000 genes) |
| 5 | `02_model_concatenated.ipynb` | Naive concatenation baseline (10,000 features) |
| 6 | `03_model_mofa.ipynb` | MOFA latent factors (15 factors) |
| 7 | `04_compare_results.ipynb` | 3-way comparison + plots |
| 8 | `05_statistical_significance.ipynb` | Paired Wilcoxon tests across folds |

## Models

Elastic-Net Cox, Random Survival Forest, XGBoost-Cox. Same folds, same metrics (C-index, IBS, dynamic AUC), for each of the three input representations above.

## Key results

- No representation shows a statistically significant advantage on C-index or IBS (paired Wilcoxon, 5 folds).
- MOFA's real advantage: numerical stability (0/5 unstable folds vs. up to 3/5 for the others) and ~10x faster runtime.

## Known limitation

MOFA factors were fit once on the full sample pool before cross-validation splitting, not refit per fold. Unsupervised (doesn't use survival label), so this is mild leakage, not outcome leakage, but a fully rigorous version would refit MOFA per training fold.

## Requirements

```
pip install pandas numpy scikit-learn scikit-survival xgboost seaborn matplotlib scipy --break-system-packages
```

## Data source

TCGA-KIRC via [Xena Browser](https://xenabrowser.net/datapages/?cohort=TCGA%20Kidney%20Clear%20Cell%20Carcinoma%20(KIRC))
