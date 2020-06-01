import pandas as pd
import numpy as np
from utils.Plot import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib as plt
path1="../Data/outputs/pred-bnn-mrna.csv"
path2="../Data/outputs/pred-bnn-meth.csv"
path3="../Data/outputs/pred-bnn-micro-mrna.csv"

data1 = pd.read_csv(path1).drop(columns=["Unnamed: 0"])
data2 = pd.read_csv(path2).drop(columns=["Unnamed: 0"])
data3 = pd.read_csv(path3).drop(columns=["Unnamed: 0"])
y_pred=np.argmax(np.maximum(data1,np.maximum(data2,data3)).values,axis=1)
true_path="../Data/outputs/true-labels.csv"
target= pd.read_csv(true_path).drop(columns=["Unnamed: 0"])
print("")
names=["a","b","c","d","e"]
cnf_matrix = confusion_matrix(target.drop(0).values,y_pred)
np.set_printoptions(precision=2)
# PlotDir non-normalized confusion matrix
plt.figure.Figure(figsize=(10, 10))
plot_confusion_matrix(cnf_matrix,
                      title="mlp finale",classes=names)
print(classification_report(target.drop(0), y_pred, ))
