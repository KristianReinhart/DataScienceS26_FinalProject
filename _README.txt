This README contains important notes about the execution order or special considerations for the scripts / code.

1) 00_Dataset_Download_KIRC.Rmd
Run all code chunks using RStudio.
Creates a subfolder "TCGA-KIRC" in the execution directory and downloads, unpacks and renames all relevant TCGA KIRC omics layers.

2) 01_QC.Rmd
Run all code chunks using RStudio.
Contains quality control plots for all relevant TCGA KIRC omics layers.

3) 02_EDA_KIRC.Rmd
Run all code chunks using RStudio.
Contains exploratory data analysis.

4) 03_DGE_KIRC.Rmd
Run all code chunks using RStudio.
Contains differential gene expression analysis.



MOFA AND ANNOTATION OF MOFA RESULTS:
The code and working directory is supposed to be in the neighboring Directory to the data Directory:
--Group_7
 --TCGA-KIRC
 --MOFA
  --Code.ipynb
  --Code.rmd

This code uses Jupyter lab, meaning once opened, the working directory Needs to be set at the beginning of each Notebook. There is a Dedicated codeblock at the top for each Notebook to set the working Directory. There are also rmd files, that also have a codeblock at the beginning to set the working directory. Both Mofa ipynb and rmd files are supposed to work from the MOFA directory.

Setting up the Environment for MOFA in jupyter lab from the cmd terminal on windows:
conda create -n mofa_env python=3.10 -y
conda activate mofa_env

# Install core scientific stack via conda-forge
conda install -c conda-forge numpy=1.26.4 scipy=1.11.4 scikit-learn=1.3.2 umap-learn ipykernel -y

# Install mofapy2, h5py and, if necessary, jupyter lab via pip:
pip install h5py mofapy2 jupyterlab

#Install statistical Tools for the analyses(you probably have those, but in case you use a new python Version):
pip install os pandas numpy seaborn matplotlib scipy  lifelines


jupyter lab


FILES:

MOFA Training and Analysis files:
The Mofa files generally require the trained model to run, but have the results already showing in their current uploaded form. If you want to rerun the code, you can either retrain the MOFA model, which takes Hours, or ask us to upload the trained model. We haven't uploaded it currently due to space, and it wasn't part of the required uploaded data.

MOFA_preperation_and_training.ipynb
Run this file ONLY, if you plan to train your own model!!
When you do, the code is seperated into codeblocks.
1.Set working Directory
2.Create the filtered methylation table
3.Train the model mofa_kidney_filtered_model.hdf5,(DO NOT USE, IT TAKES HOURS!) which uses RNAseq, Methylation and RPPA.

MOFA_Analysis_and_exports.ipynb
This file only works, if mofa_kidney_filtered_model.hdf5 is in the same working Directory. This file goes over the already trained model and Shows the explained variance of the factors, the significance of phenotype to clinical association and  reaturns the most influential factors for the Annotation and GO Enrichment. There aren't computationally heavy computations Happening here, so the entire file can just be run.
1.Set working Directory
2.Get explained variance of each cohort per factor across all views
3.Get top driving features per factor
4.Analysis of association between factors and clinical/phenotype data
5.Extraction of top 500 Features per factor per view (data/mofa_top_500_features_per_factor.csv). This is used for Annotation and ML, so this Output file is important.
6.Plotting of the variance explanation of the cohorts by the factors per view, Plots are viewable under mofa_plots/mofa_variance_Cohort_view.png
7.Plotting of Clinical/Phenotype association significance mofa_plots\mofa_factors_phenotype_associations.png
8.Comparisons of normal vs disease cohort variance Level per factor
9.Plotting of normal vs disease cohort variance Level per factor
10.Z-matrix Output to data/mofa_sample_factor_matrix.csv, used for ML

Mofa_2_ALL_data_training.ipynb
This file trains the MOFA using all 5 layers with no Filtration. This model has an imputed RPPA layer due to learned new latent factors (Z) that captured major biological variance present in the Normal Group.
1.Set working Directory
2.Train Model(DO NOT USE, IT TAKES HOURS! mofa_tcga_kirc_5layers_model.hdf5 can be uploaded by request!
3.Get explained variance of each cohort per factor across all views
4.Get top driving features per factor
5.Analysis of association between factors and clinical/phenotype data
6.Extraction of top 100 Features per factor per view(mofa_top_100_features_per_factor_all_data.csv), isn't used anymore
7.Plotting of the variance explanation of the cohorts by the factors per view, Plots are viewable under mofa_plots/mofa_variance_all_data_Cohort_view.png
8.Plotting of Clinical/Phenotype association significance

Mofa_2_only_CNV_excluded.ipynb
This file trains the MOFA using 4 layers with no Filtration. It was trained to test whether it also imputes the RPPA normal cohort W weight Matrix, that has Variance Explanation despite there being no normal RPPA samples. It imputed the RPPA with 62% variance explained, which is the same result as with all 5 layers, so we can say that it imputes the normal layer due to the presence of the exon and RNA data.
1.Set working Directory
2.Train Model(DO NOT USE, IT TAKES HOURS! mofa_tcga_kirc_5layers_model_no_cnv.hdf5 can be uploaded by request!
3.Get explained variance of each cohort per factor across all views
4.Get top driving features per factor
5.Analysis of association between factors and clinical/phenotype data
6.Extraction of top 100 Features per factor per view(mofa_top_100_features_per_factor_all_data.csv), isn't used
7.Plotting of the variance explanation of the cohorts by the factors per view, Plots are viewable under mofa_plots/mofa_variance_all_data_Cohort_view.png
8.Plotting of Clinical/Phenotype association significance

R Annotation files:
The important file to have is the data/mofa_top_500_features_per_factor.csv. without that file, the code doesn't work. The files are Rmarkdown files, that again have the working Directory set in the beginning. Aside from that, you can run all chunks without issues. The longest part is the Annotation and pathway significance, taking a few minutes.

meth_anno.Rmd
The Goal of this file is to perform GO Enrichment of the top 500 Features of all 15 factors to find significant pathways.
1.Set working Directory
2.Package installing and loading (automatically checks what needs to be installed)
3.Load and check the methylation Features, requires mofa_top_500_features_per_factor.csv
4.Annotation of the methylation labels using IlluminaHumanMethylation450kanno.ilmn12.hg19
5.Analysis of the significance of the CPGs using missMethyl GO Enrichment and KEGG per factor
6.Creation of summary file of all results for later
7.Sorting of the pathways by significance and concatination into final Output Significant_missMethyl_GO_FDR_0_05.csv
8.Bubbleplot of sigificant pathways

RNA_annotation.Rmd
The Goal of this file is to perform GO Enrichment of the top 500 Features of all 15 factors to find significant pathways.
1.Set working Directory
2.Package installing and loading (automatically checks what needs to be installed)
3.Load and check the RNA Features, requires mofa_top_500_features_per_factor.csv
4.Annotation of the RNA labels using org.Hs.eg.db from the human Annotation database
5.Analysis of the significance of the Genes using GO Enrichment enrichGO() per factor with cutoff of FDR<0.05
6.Creation of summary file of all results into All_Factors_Significant_GO.csv
7.Bubbleplot of sigificant pathways
8.Top 10 pathways per factor are stored to Top_Pathway_Per_Factor.csv

Top annotation plots RNA_METH.Rmd
This file is used to create some barplots, and requires Top_Pathway_Per_Factor.csv and Significant_missMethyl_GO_FDR_0_05.csv to function.
1.Set working Directory
2.Create Barplots to mofa_plots\Pathway_Barplots view_Top_Annotation_Per_Factor.png

