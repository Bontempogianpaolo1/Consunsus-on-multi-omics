from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import matplotlib as plt
from utils.Plot import plot_confusion_matrix
from utils.Plot import plot_confusion_matrix
from utils.Plot import plot_confusion_matrix
seed=1200
annotation_path="../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names=y.unique()
y=y.astype('category').cat.codes


meth_path="../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path="../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path="../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files=[meth_path,mRNA_path,mRNA_normalized_path]
filenames=["meth","mrna","mrna normalized"]
for file,filename in zip(files,filenames):
    X= pd.read_csv(file).drop(columns=["Composite Element REF","Unnamed: 0"])
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=seed)
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    print(filename)
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
                          title=filename,classes=names)
    # PlotDir normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix,  normalize=True,
                          title="normalizeD "+filename,classes=names)
    print(classification_report(y_test, y_pred, ))


