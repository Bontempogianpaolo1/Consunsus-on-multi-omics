import pandas as pd
import numpy as np
from utils.Plot import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib as plt
path1="../Data/outputs/pred-mlp-mrna.csv"
path2="../Data/outputs/pred-mlp-meth.csv"
path3="../Data/outputs/pred-mlp-micro mrna.csv"
annotation_path="../Data/data/preprocessed_annotation_global.csv"
data1 = pd.read_csv(path1).drop(columns=["Unnamed: 0"])
data2 = pd.read_csv(path2).drop(columns=["Unnamed: 0"])
data3 = pd.read_csv(path3).drop(columns=["Unnamed: 0"])
model="mlp"
filename= ["mrna","meth","micro mrna"]
with open('../Data/outputs/bnn mrna.txt', 'w') as f:
    y_pred=np.argmax(data1.values,axis=1)
    true_path="../Data/outputs/true-labels.csv"
    target= pd.read_csv(true_path).drop(columns=["Unnamed: 0"])
    names=pd.read_csv(annotation_path)["label"].astype('category').cat.categories
    cnf_matrix = confusion_matrix(target.drop(0).values,y_pred)
    np.set_printoptions(precision=2)
    # PlotDir non-normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix,title=model+"-"+filename[0],classes=names)
    print(classification_report(target.drop(0), y_pred, ),file=f)
with open('../Data/outputs/bnn meth.txt', 'w') as f:
    y_pred=np.argmax(data2.values,axis=1)
    true_path="../Data/outputs/true-labels.csv"
    target= pd.read_csv(true_path).drop(columns=["Unnamed: 0"])
    names=pd.read_csv(annotation_path)["label"].astype('category').cat.categories
    cnf_matrix = confusion_matrix(target.drop(0).values,y_pred)
    np.set_printoptions(precision=2)
    # PlotDir non-normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix,title=model+"-"+filename[1],classes=names)
    print(classification_report(target.drop(0), y_pred, ),file=f)
with open('../Data/outputs/bnn-micro-rna.txt', 'w') as f:
    y_pred=np.argmax(data3.values,axis=1)
    true_path="../Data/outputs/true-labels.csv"
    target= pd.read_csv(true_path).drop(columns=["Unnamed: 0"])
    names=pd.read_csv(annotation_path)["label"].astype('category').cat.categories
    cnf_matrix = confusion_matrix(target.drop(0).values,y_pred)
    np.set_printoptions(precision=2)
    # PlotDir non-normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix,title=model+"-"+filename[2],classes=names)
    print(classification_report(target.drop(0), y_pred, ),file=f)


with open('../Data/outputs/bnn comparison.txt', 'w') as f:
    y_pred=np.argmax(np.maximum(data1,np.maximum(data2,data3)).values,axis=1)
    true_path="../Data/outputs/true-labels.csv"
    target= pd.read_csv(true_path).drop(columns=["Unnamed: 0"])
    names=pd.read_csv(annotation_path)["label"].astype('category').cat.categories
    cnf_matrix = confusion_matrix(target.drop(0).values,y_pred)
    np.set_printoptions(precision=2)
    # PlotDir non-normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix,title="comparisonmlp",classes=names)
    print(classification_report(target.drop(0), y_pred, ),file=f)
