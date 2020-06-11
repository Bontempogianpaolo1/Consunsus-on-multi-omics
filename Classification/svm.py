import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from utils.Plot import plot_confusion_matrix

seed = 1200
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names = y.astype('category').cat.categories
y = y.astype('category').cat.codes

modelname = "svm"
meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path, mRNA_path, mRNA_normalized_path]
filenames = ["meth", "mrna", "micro mrna"]
predictions = []
true_labels = []
parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': [1, 10, 20], 'random_state': [seed]}

for file, filename in zip(files, filenames):
    outputname = modelname + filename
    with open('../Data/outputs/' + outputname + '.txt', 'w') as f:
        X = pd.read_csv(file, index_col=False, header=None)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)
        model = svm.SVC()
        model = GridSearchCV(model, parameters)
        pca = PCA(n_components=7)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_test_transformed)
        predictions.append(y_pred)
        true_labels.append(y_test)
        print(filename)
        print("best parameters")
        # print(model.best_params_)
        print("Confusion matrix")
        totalscore = accuracy_score(y_test, y_pred)
        print("final score : %f" % totalscore)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(10, 10))
        # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
        print()
        np.set_printoptions(precision=2)
        # PlotDir non-normalized confusion matrix
        plt.figure.Figure(figsize=(10, 10))
        plot_confusion_matrix(cnf_matrix,
                              title=modelname + "-" + filename, classes=names)

        print(modelname + filename + " " + str(model.best_params_), file=f)
        print(classification_report(y_test, y_pred, ), file=f)
names = np.append(names, "unknown")
unknown_index = np.logical_not(
    np.logical_or(
        predictions[0] == predictions[1],
        np.logical_or(
            predictions[0] == predictions[2],
            predictions[1] == predictions[2]
        )
    )
)

y_pred[predictions[0] == predictions[1]] = predictions[0][predictions[0] == predictions[1]]
y_pred[predictions[0] == predictions[2]] = predictions[0][predictions[0] == predictions[2]]
y_pred[predictions[1] == predictions[2]] = predictions[1][predictions[1] == predictions[2]]
cnf_matrix = confusion_matrix(true_labels[0], y_pred)
# plt.figure(figsize=(10, 10))
# plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
print()
np.set_printoptions(precision=2)
# PlotDir non-normalized confusion matrix
plt.figure.Figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix,
                      title="comparison" + modelname, classes=names)

with open('../Data/outputs/' + modelname + '.txt', 'w') as f:
    print(classification_report(true_labels[0], y_pred, ), file=f)
