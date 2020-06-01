from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import matplotlib as plt
from utils.Plot import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
seed=1200
annotation_path="../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names=y.astype('category').cat.categories
y=y.astype('category').cat.codes


meth_path="../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path="../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path="../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files=[meth_path,mRNA_path,mRNA_normalized_path]
filenames=["meth","mrna","mrna normalized"]
parameters = { 'criterion':["gini", "entropy"], 'max_depth':[5,10,15],'min_samples_split': [2,4,10]}

predictions=[]
true_labels=[]
for file,filename in zip(files,filenames):
    X= pd.read_csv(file).drop(columns=["Composite Element REF","Unnamed: 0"])
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=seed,stratify=y)
    model = RandomForestClassifier()
    model = GridSearchCV(model, parameters)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    predictions.append(y_pred)
    true_labels.append(y_test)
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
                          title="random forest "+filename,classes=names)
    # PlotDir normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix,  normalize=True,
                          title="normalizeD "+filename,classes=names)
    print("random forest "+filename)
    print(classification_report(y_test, y_pred, ))
names=np.append(names,"unknown")
unknown_index = np.logical_not(np.logical_or(predictions[0]==predictions[1],np.logical_or(predictions[0]==predictions[2],predictions[1]==predictions[2])))

y_pred=predictions[0].copy()
y_pred[unknown_index] = 5
y_pred[predictions[0]==predictions[1]]=predictions[0][predictions[0]==predictions[1]]
y_pred[predictions[0]==predictions[2]]=predictions[0][predictions[0]==predictions[2]]
y_pred[predictions[1]==predictions[2]]=predictions[1][predictions[1]==predictions[2]]
cnf_matrix = confusion_matrix(true_labels[0], y_pred)
# plt.figure(figsize=(10, 10))
# plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
print()
np.set_printoptions(precision=2)
# PlotDir non-normalized confusion matrix
plt.figure.Figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix,
                      title="mlp finale",classes=names)
print(classification_report(true_labels[0], y_pred, ))





