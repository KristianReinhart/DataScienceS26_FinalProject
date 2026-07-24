# Multi-Omics Integration for Survival Prediction in Clear Cell Renal Cell Carcinoma (TCGA-KIRC)

**Group 7** — Maksim Danilchyk, Sofya Shorzhina, Tilo Alves Radtke, Kristian Reinhart
Data Science SoSe 2026 — Final Project

---

## Research question

Does unsupervised multi-omics integration with MOFA produce latent factors that improve
survival prediction in clear cell renal cell carcinoma (ccRCC), compared with survival models
trained on raw single-omics features or on a naive concatenation of several omics layers?

We answer this in five stages: data download, quality control, exploratory analysis,
differential expression, multi-omics factor analysis (MOFA), and a survival machine-learning
benchmark that compares three feature representations on identical cross-validation folds.

---

## 1. Repository structure

```
DataScienceS26_FinalProject-main/
├── README.md                            <- this file
├── 00_Dataset_Download_KIRC.Rmd         <- downloads all omics layers into TCGA-KIRC/
├── 01_QC.Rmd                            <- per-layer quality control
├── 02_EDA_KIRC.Rmd                      <- exploratory data analysis
├── 03_DGE_KIRC.Rmd                      <- differential gene expression (tumour vs normal)
│
├── MOFA/
│   ├── MOFA_preperation_and_training.ipynb   <- filters methylation, trains main model
│   ├── MOFA_Analysis_and_exports.ipynb       <- factor analysis + exports (REQUIRED for ML)
│   ├── Mofa_2_ALL_data_training.ipynb        <- 5-layer model (supplementary)
│   ├── Mofa_2_no_CNV.ipynb                   <- 4-layer model (supplementary)
│   ├── RNA_annotation.Rmd                    <- GO enrichment of RNA factor loadings
│   ├── meth_annotation.Rmd                   <- GO/KEGG enrichment of methylation loadings
│   └── Top annotation plots RNA_METH.Rmd     <- summary barplots
│
└── MLPipeline/
    ├── 00_preprocessing_kirc_multiomics_v2.ipynb  <- filter/impute RNA + methylation
    ├── 00_build_reference_cohort.ipynb            <- fixed cohort + 5 CV folds
    ├── common_utils.py                            <- shared modelling code + CONFIG
    ├── 01_model_expression.ipynb                  <- single-omics baseline
    ├── 02_model_concatenated.ipynb                <- naive concatenation baseline
    ├── 03_model_mofa.ipynb                        <- MOFA-factor model
    ├── 04_compare_results_1.ipynb                 <- 3-way comparison + plots
    └── 05_statistical_significance.ipynb          <- paired Wilcoxon tests
```

---

## 2. Requirements

### 2.1 R (tested on R 4.6.0, RStudio)

Install CRAN and Bioconductor packages once:

```r
install.packages(c(
  "R.utils", "data.table", "tidyverse", "matrixStats", "UpSetR", "pheatmap",
  "corrplot", "survival", "survminer", "umap", "ggrepel", "patchwork",
  "RColorBrewer", "readr", "proxy", "circlize", "knitr", "rmarkdown"
))

install.packages("BiocManager")
BiocManager::install(c(
  "limma", "clusterProfiler", "org.Hs.eg.db", "fgsea", "msigdbr",
  "AnnotationDbi", "ComplexHeatmap", "vsn", "missMethyl",
  "IlluminaHumanMethylation450kanno.ilmn12.hg19"
))
```

The annotation Rmd files in `MOFA/` check for missing packages and install them automatically.

### 2.2 Python (tested with Python 3.10, Jupyter Lab)

```bash
conda create -n mofa_env python=3.10 -y
conda activate mofa_env
conda install -c conda-forge numpy=1.26.4 scipy=1.11.4 scikit-learn=1.3.2 umap-learn ipykernel -y
pip install h5py mofapy2 jupyterlab
pip install pandas seaborn matplotlib lifelines
pip install scikit-survival xgboost
jupyter lab
```

`mofapy2` is required only for MOFA training and loading. `scikit-survival` and `xgboost` are
required only for the ML pipeline.

### 2.3 Disk and time

The full download is roughly **2.7 GB**. The methylation matrix alone is ~1.4 GB and needs
about 3–4 GB of RAM to load. Expect the following approximate runtimes:

| Stage | Runtime (estimated) |
|---|---|
| 00 Download | 5–20 min (network dependent) |
| 01 QC | 3–5 min |
| 02 EDA | 5–10 min (methylation load dominates) |
| 03 DGE | 2–4 min |
| MOFA training | **several hours** — see §6 |
| MOFA analysis + annotation | 5–15 min |
| ML pipeline | 20 min – several hours depending on `debug_mode` |

---

## 3. Required directory layout

All scripts assume the data folder `TCGA-KIRC/` sits at the **repository root**, next to the
numbered Rmd files:

```
DataScienceS26_FinalProject-main/
├── TCGA-KIRC/          <- created by 00_Dataset_Download_KIRC.Rmd
├── 00..03_*.Rmd
├── MOFA/
└── MLPipeline/
```

The R Markdown files at the root resolve data through the `data_dir: "TCGA-KIRC"` parameter in
their YAML header, so they work as long as you knit them from the repository root.

The Jupyter notebooks in `MOFA/` and `MLPipeline/` **do not** discover their location
automatically. Each has a dedicated first cell containing `os.chdir(...)` (or `setwd(...)` in
the Rmd files) that you must edit before running. See §7 for the exact values.

---

## 4. Reproduction order

Run the stages strictly in this order. Later stages may need files produced by earlier ones.

| Step | File | Produces |
|---|---|---|
| 1 | `00_Dataset_Download_KIRC.Rmd` | `TCGA-KIRC/` with 9 data files |
| 2 | `01_QC.Rmd` | QC report |
| 3 | `02_EDA_KIRC.Rmd` | EDA report |
| 4 | `03_DGE_KIRC.Rmd` | DGE report + `DGE_tumor_vs_normal_limma.csv` |
| 5 | `MOFA/MOFA_preperation_and_training.ipynb` | filtered methylation + trained model (`.hdf5`) |
| 6 | `MOFA/MOFA_Analysis_and_exports.ipynb` | `data/mofa_top_500_features_per_factor.csv`, `data/mofa_sample_factor_matrix.csv` |
| 7 | `MOFA/RNA_annotation.Rmd`, `MOFA/meth_annotation.Rmd` | GO enrichment tables |
| 8 | `MOFA/Top annotation plots RNA_METH.Rmd` | summary barplots |
| 9 | `MLPipeline/00_preprocessing_kirc_multiomics_v2.ipynb` | `preprocessed_multiomics/` |
| 10 | `MLPipeline/00_build_reference_cohort.ipynb` | `reference_cohort/` (cohort + CV folds) |
| 11 | `MLPipeline/01_model_expression.ipynb` | `results/results_expression.pkl` |
| 12 | `MLPipeline/02_model_concatenated.ipynb` | `results/results_concatenated.pkl` |
| 13 | `MLPipeline/03_model_mofa.ipynb` | `results/results_mofa.pkl` |
| 14 | `MLPipeline/04_compare_results_1.ipynb` | comparison plots + `final_comparison_summary.csv` |
| 15 | `MLPipeline/05_statistical_significance.ipynb` | `statistical_significance_results.csv` |

Steps 5–8 (MOFA) and steps 9–15 (ML) both depend on step 1 only, so they can be run in
parallel — **except** that step 13 needs the factor matrix from step 6.

---

## 5. Stage details

### Stage 0–3: R notebook analyses

Open the project in RStudio, set the working directory to the repository root, and knit each
file in order (`Knit` button, or `rmarkdown::render("02_EDA_KIRC.Rmd")`).

- **`00_Dataset_Download_KIRC.Rmd`** creates `TCGA-KIRC/` and downloads, unpacks and renames
  nine files from the UCSC Xena S3 mirror: `HiSeqV2.txt` (RNA-seq),
  `HiSeqV2_exon.txt` (exon expression), `HumanMethylation450.txt` (methylation),
  `Gistic2_CopyNumber_Gistic2_all_data_by_genes.txt` and
  `..._all_thresholded.by_genes.txt` (copy number), `KIRC_mc3_gene_level.txt` (mutations),
  `RPPA.txt` (protein), `KIRC_clinicalMatrix.txt` (phenotypes) and `KIRC_survival.txt`
  (curated survival). Downloads have a 10-minute timeout each and fail with a
  warning, so check the console output or the TCGA-KIRC subdirectory for 9 extracted .txt files
  before continuing.
- **`01_QC.Rmd`** produces per-layer quality control for all eight layers.
- **`02_EDA_KIRC.Rmd`** covers sample inventory and cross-layer overlap, per-layer QC,
  clinical and survival description, removal of a sex-driven technical axis in the methylation
  data, PCA/UMAP structure and a cross-modal comparison.
- **`03_DGE_KIRC.Rmd`** runs the tumour-versus-normal differential expression analysis with
  `limma`, functional enrichment (GO and Hallmark GSEA), an immune-infiltrate control, and a
  Cox bridge linking differential expression to survival. It writes
  `DGE_tumor_vs_normal_limma.csv`.

Both `02` and `03` are parameterised in their YAML header (`data_dir`, thresholds, seed) and
cache intermediate results. If a knit fails partway through, delete the `*_cache/` folder
before re-knitting — an interrupted knit can leave a corrupt cache entry.

### Stage 4: MOFA

Run from the `MOFA/` directory in Jupyter Lab, setting the working directory in the first cell
of each notebook.

- **`MOFA_preperation_and_training.ipynb`** builds the filtered methylation table
  (`data/HumanMethylation450_filtered.csv`) and trains the main model
  (`mofa_kidney_filtered_model.hdf5`) on RNA-seq, methylation and RPPA.
  **Training takes several hours.** Run this only if you intend to retrain.
- **`MOFA_Analysis_and_exports.ipynb`** is the important one. It loads the trained model and
  produces the explained-variance breakdown, factor-to-phenotype association tests, the top
  500 features per factor (`data/mofa_top_500_features_per_factor.csv`) and the sample-by-factor
  matrix (`data/mofa_sample_factor_matrix.csv`). The last two files feed the annotation and ML
  stages, so this notebook must be run.
- **`Mofa_2_ALL_data_training.ipynb`** and **`Mofa_2_no_CNV.ipynb`** are supplementary models
  (five layers, and four layers without CNV) used to test whether MOFA imputes the missing
  normal-tissue RPPA layer. They are not required to reproduce the main results.
- **`RNA_annotation.Rmd`** and **`meth_annotation.Rmd`** perform GO (and KEGG) enrichment on
  the top 500 features per factor. Both require
  `data/mofa_top_500_features_per_factor.csv`. **`Top annotation plots RNA_METH.Rmd`** then
  builds summary barplots from their outputs.

### Stage 5: ML pipeline

Run from `MLPipeline/` in Jupyter Lab. `common_utils.py` holds the shared modelling code and a
single `CONFIG` dictionary controlling folds, endpoint (`OS`), random seed (42) and model
hyperparameters.

The benchmark compares three feature representations on **identical stratified folds**:

1. **Single-omics** — variance-filtered gene expression.
2. **Naive concatenation** — standardised expression + methylation features.
3. **MOFA factors** — the latent factors from stage 4.

Each representation is evaluated with Elastic-Net Cox, Random Survival Forest and
XGBoost-Cox, scored by concordance index, integrated Brier score and time-dependent AUC.
`04_compare_results_1.ipynb` builds the comparison; `05_statistical_significance.ipynb` runs
paired Wilcoxon tests across folds.

---

## 6. Trained MOFA models are not included

The trained models (`mofa_kidney_filtered_model.hdf5` and the two supplementary models) are
**not in this repository** because of file-size limits. You have two options:

1. **Retrain** by running `MOFA_preperation_and_training.ipynb`. This takes several hours.
2. **Request the trained model files from the authors** and place them in `MOFA/`.

`MOFA_Analysis_and_exports.ipynb` cannot run without `mofa_kidney_filtered_model.hdf5` in the
same directory. The notebooks in the repository are saved with their outputs intact, so the
results remain readable without rerunning them.

---

## 7. Running the notebooks: working directories

**Start every Jupyter notebook from the folder it lives in.** JupyterLab starts the kernel in
the notebook's own directory, so opening `MLPipeline/01_model_expression.ipynb` from the file
browser is enough. The notebooks resolve everything from there:

- `MLPipeline/*` reads the raw data through `../TCGA-KIRC/`, writes
  `preprocessed_multiomics/`, `reference_cohort/` and `results/` inside `MLPipeline/`, and
  reads the MOFA exports through `../MOFA/data/`.
- `MOFA/*` reads the raw data through `../TCGA-KIRC/` and writes `data/` and `mofa_plots/`
  inside `MOFA/`.

**Knit the R notebook files from their own folder** as well. The four numbered files at the
repository root resolve data through their `data_dir: "TCGA-KIRC"` parameter. The three
annotation files in `MOFA/` contain a commented-out `setwd()` line; knitr sets the working
directory automatically when knitting, so uncomment it only if you execute chunks
interactively from a different location.

**One setting worth knowing about.** `MLPipeline/common_utils.py` contains a `debug_mode`
switch, and notebooks `01`–`04` set it explicitly at the top. It is `False` everywhere, which
is the full-scale configuration used to produce the reported results. Setting it to `True`
subsamples to 60 samples, 200 features, 20 trees and 3 folds; that is useful for a quick
test but does not reproduce the results. If you use it, set it consistently across
notebooks `01`, `02` and `03`, otherwise the three representations are no longer compared on
equal terms.

---

## 8. Known limitations

- **MOFA factors were fit once on the full sample pool** before the cross-validation split,
  rather than refit inside each training fold. MOFA is unsupervised and never sees the survival
  labels, so this is mild feature leakage rather than outcome leakage, but a fully rigorous
  version would refit MOFA per fold.
- **No representation shows a statistically significant advantage** on concordance index or
  integrated Brier score across the five folds (paired Wilcoxon). The measurable advantage of
  the MOFA representation is numerical stability (0 of 5 unstable folds, against up to 3 of 5
  for the alternatives) and roughly tenfold faster runtime.
- The tumour-versus-normal contrast in `03_DGE_KIRC.Rmd` is confounded with cell-type
  composition; the notebook quantifies this with an immune-infiltrate score and interprets the
  enrichment accordingly.

---

## 9. Data source

TCGA Kidney Renal Clear Cell Carcinoma (KIRC), obtained from the UCSC Xena browser:
<https://xenabrowser.net/datapages/?cohort=TCGA%20Kidney%20Clear%20Cell%20Carcinoma%20(KIRC)>

All layers are Level-3 processed data redistributed by UCSC Xena. Curated survival endpoints
come from the TCGA Pan-Cancer Clinical Data Resource (Liu et al., *Cell* 2018). Sample-type
codes follow the GDC barcode specification, where `01` marks a primary tumour, `05` an
additional new primary and `11` solid-tissue normal.

No access restrictions apply; all data is public.

---

## 10. Authors

Group 7 — Maksim Danilchyk, Sofya Shorzhina, Tilo Alves Radtke, Kristian Reinhart.

This README consolidates and supersedes the earlier `MOFA/README.txt` and
`MLPipeline/README_ML.md`, which are retained for reference.
