# bnn_on_multi_omics
Due to the huge amount of available omic data, classifying samples according to various omics is a complex process. One of the most common approaches consists of creating a classifier for each omic and subsequently making a consensus among the classifiers that assigns to each sample the most voted class among the outputs on the individual omics. 
 
However, this approach does not consider the confidence in the prediction ignoring that a biological information coming from a certain omic may be more reliable than others.
Therefore, it is here proposed a method consisting of a tree-based multi-layer perceptron (MLP), which estimates the class-membership probabilities for classification. In this way, it is not only possible to give relevance to all the omics, but also to label as \textit{Unknown} those samples for which the classifier is uncertain in its prediction.\\
The method was applied to a dataset composed of 909 kidney cancer samples for which these three omics were available: gene expression (mRNA), microRNA expression (miRNA) and methylation profiles (meth) data. The method is valid also for other tissues and on other omics (e.g. proteomics, copy number alterations data, single nucleotide polymorphism data).
This tool can therefore be particularly useful in clinical practice, allowing physicians to focus on the most interesting and challenging samples.



# Setup

Data: for privacy reasons the data are not available. For requests leave a comment.

After the download run the Data/Anomalies_Data_normalize.py to normalize and prepocess data.
To Have a visualization of the data run Data/pca_visualization.ipynb

To obtain several confusion matrices on different omics using mlp,BNN, and MLPTREE run Classification/outliers.py
To obtain several confusion matrices on different omics using SVM and Random Forest  run Classification/random_forest.py and Classification/svm.py
To obtain several confusion matrices on consunsus using mlp,BNN, and MLPTREE run Classification/plot_comparison-new.py(Classification/outliers.py must run first)

