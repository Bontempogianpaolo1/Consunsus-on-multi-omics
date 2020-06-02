from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import matplotlib as plt
from utils.Plot import plot_confusion_matrix
from utils.Plot import plot_confusion_matrix
from utils.Plot import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
seed=1200
annotation_path="../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names=y.astype('category').cat.categories
y=y.astype('category').cat.codes

modelname=" mlp "
meth_path="../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path="../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path="../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files=[meth_path,mRNA_path,mRNA_normalized_path]
filenames=["meth","mrna","mrna normalized"]
predictions=[]
parameters = { 'hidden_layer_sizes':np.arange(start=100,stop= 150,step=10), 'random_state':[seed],'max_iter': [200,400,600 ]}

true_labels=[]
for file,filename in zip(files,filenames):
    outputname=modelname+filename
    with open('../Data/outputs/' + outputname + '.txt', 'w') as f:
        X= pd.read_csv(file).drop(columns=["Composite Element REF","Unnamed: 0"])
        X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=seed,stratify=y)
        model = MLPClassifier()
        model= GridSearchCV(model,parameters)
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        predictions.append(y_pred)
        true_labels.append(y_test)
        print(filename)
        print("best parameters")
        #print(model.best_params_)
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
                              title=modelname+filename,classes=names)

        print(modelname+filename+" "+str(model.best_params_),file=f)
        print(classification_report(y_test, y_pred, ),file=f)
names=np.append(names,"unknown")
unknown_index = np.logical_not(
    np.logical_or(
        predictions[0]==predictions[1],
        np.logical_or(
            predictions[0]==predictions[2],
            predictions[1]==predictions[2]
        )
    )
)

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
                      title="comparison"+modelname,classes=names)
plt.pyplot.savefig(modelname+".png")

with open('../Data/outputs/'+modelname+'.txt', 'w') as f:
    print(classification_report(true_labels[0], y_pred, ),file=f)
