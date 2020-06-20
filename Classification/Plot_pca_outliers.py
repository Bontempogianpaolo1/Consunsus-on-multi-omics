import matplotlib as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import Classification.Mlp as mlp
import Classification.Mlptree as mlptree
import Classification.bnn as bnn
from utils.Plot import plot_confusion_matrix
from utils.Plot import plot_outliers

seed = 1200
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)
# y = pd.read_csv(annotation_path)["label"]
# names = y.astype('category').cat.categories
# y = y.astype('category').cat.codes
modelname = " mlp "
meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path,  mRNA_normalized_path,mRNA_path]
filenames = ["micro mrna", "meth", "mrna"]
parameters = {'hidden_layer_sizes': np.arange(start=100, stop=150, step=10), 'random_state': [seed],
              'max_iter': [200, 400, 600]}
true_labels = []
for file, filename in zip(files, filenames):
    outputname = modelname + filename
    X = pd.read_csv(file, index_col=False, header=None)
    y2 = pd.read_csv("../Data/data/anomalies_preprocessed_annotation_global.csv")
    X2 = pd.read_csv("../Data/data/anomalies_preprocessed_Matrix_" + filename + ".csv", index_col=False, header=None)
    if filename == "mrna":
        X = pd.DataFrame(X[X.std().sort_values(ascending=False).head(1200).index].values.tolist())
        X2 = pd.DataFrame(X2[X2.std().sort_values(ascending=False).head(1200).index].values.tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed,
                                                        stratify=y["label"].astype('category').cat.codes)

    models = [bnn.BNN, mlp.MLP, mlptree.MlpTree]
    modelnames = [ "bnn","mlptree", "mlp"]
    names = y["label"].astype('category').cat.categories
    names2=names.append(pd.Index(["Unknown"]))
    n_components = 7
    pca = PCA(n_components=n_components)

    X_train_transformed = pca.fit_transform(X_train)
    X_test_transformed = pca.transform(X_test)
    X_transformed = pca.transform(X2)
    for model, modelname in zip(models, modelnames):
        # clf=model
        y_pred = pd.read_csv("../Data/outputs/pred-testset-" + modelname + "-" + filename + ".csv")['y_pred']

        y_pred = y_pred.astype(np.float)
        y_true = y_test['label'].astype('category').cat.codes
        y_true[y_true != 5] = 1
        y_true[y_true == 5] = -1
        y_pred[y_pred != 5] = 1
        y_pred[y_pred == 5] = -1
        plot_outliers(X_test_transformed, y_pred, X_train_transformed, "testset-newdata-" + modelname + "-" + filename + "pca")

        y_pred = pd.read_csv("../Data/outputs/pred-stomaco-" + modelname + "-" + filename + ".csv")['y_pred']
        print("plot")
        y_pred = y_pred.astype(np.float)
        y_true = y2['label'].astype('category').cat.codes
        y_true[y_true != 5] = 1
        y_true[y_true == 5] = -1
        y_pred[y_pred != 5] = 1
        y_pred[y_pred == 5] = -1
        plot_outliers(X_transformed , y_pred, X_train_transformed, "stomaco-newdata-" + modelname + "-" + filename + "pca")

