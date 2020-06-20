import matplotlib as plt
import numpy as np
import pandas as pd
import ast
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import Classification.Mlp as mlp
import Classification.Mlptree as mlptree
import Classification.bnn as bnn
from sklearn.metrics import classification_report
from utils.Plot import plot_confusion_matrix
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
names = pd.read_csv(annotation_path)["label"].astype('category').cat.categories
modelnames = [ "bnn","mlptree", "mlp"]
filenames = ["miRNA", "meth", "mRNA"]
#testset
path="../Data/outputs/pred-testset-"
for modelname in modelnames:
    data=[]
    for filename in filenames:
        X = pd.read_csv(path+ modelname + "-" + filename + ".csv")
        data.append(X)
    comparisons = np.ndarray(shape=(0,3))
    for i in range(data[1].shape[0]):
        m0 = data[0].iloc[i]
        m1 = data[1].iloc[i]
        m2 = data[2].iloc[i]
        if (m0['y_pred']==5) |  (m1['y_pred']==5) | (m2['y_pred']==5) :
            comparisons=np.vstack([comparisons,np.array([m0['official_name'],5,m0['y_true']])])
        else:
            m3=data[np.argmax([m0["max_probability"],m1["max_probability"],m2["max_probability"]])].iloc[i]
            comparisons=np.vstack([comparisons,np.array([m3['official_name'], m3['y_pred'], m3['y_true']])])


    comparisons= np.array(comparisons)

    df = pd.DataFrame({
        'official_name': comparisons[:,0].tolist(),
        'max_probability': comparisons[:,1].astype(np.float).tolist(),
        'y_pred': comparisons[:,2].astype(np.float).tolist(),
        #'y_true': comparisons[:,3].astype(np.float).tolist()
    })
    df.to_csv("../Data/outputs/pred-comparison-testset-" + modelname + "-" + filename + ".csv")
    print("plot")
    cnf_matrix = confusion_matrix(comparisons[:,2].astype(np.float), comparisons[:,1].astype(np.float))
    # plt.figure(figsize=(10, 10))
    # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
    print()
    np.set_printoptions(precision=2)
    # PlotDir non-normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))

    plot_confusion_matrix(cnf_matrix,
                          title="with-unknown-testset-" + modelname + "-comparison",classes=names)
    with open("../Data/outputs/with-unknown-testset-" + modelname + "-comparison.txt", 'w') as f:

        print(classification_report(comparisons[:,2].astype(np.float), comparisons[:,1].astype(np.float), ), file=f)

path="../Data/outputs/pred-stomaco-"
for modelname in modelnames:
    data=[]
    for filename in filenames:
        X = pd.read_csv(path+ modelname + "-" + filename + ".csv")
        data.append(X)
    #comparisons=[]
    comparisons = np.ndarray(shape=(0,3))
    for i in range(data[1].shape[0]):
        m0 = data[0].iloc[i]
        m1 = data[1].iloc[i]
        m2 = data[2].iloc[i]
        if (m0['y_pred']==5) |  (m1['y_pred']==5) | (m2['y_pred']==5) :
            comparisons=np.vstack([comparisons,np.array([m0['official_name'],5,m0['y_true']])])
        else:
            m3=data[np.argmax([m0["max_probability"],m1["max_probability"],m2["max_probability"]])].iloc[i]
            comparisons=np.vstack([comparisons,np.array([m3['official_name'], m3['y_pred'], m3['y_true']])])
    comparisons= np.array(comparisons)
    print("plot")
    y_pred=comparisons[:,1].astype(np.float)
    y_true=comparisons[:,2].astype(np.float)
    y_true[y_true != 5] = 0
    y_true[y_true == 5] = 1
    y_pred[y_pred != 5] = 0
    y_pred[y_pred == 5] = 1

    df = pd.DataFrame({
        'official_name': comparisons[:, 0].tolist(),
        'max_probability': comparisons[:, 1].astype(np.float).tolist(),
        'y_pred': comparisons[:, 2].astype(np.float).tolist(),
        #'y_true': comparisons[:, 3].astype(np.float).tolist()
    })
    df.to_csv("../Data/outputs/pred-comparison-stomaco-" + modelname + "-" + filename + ".csv")
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 10))
    # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
    print()
    np.set_printoptions(precision=2)
    # PlotDir non-normalized confusion matrix
    plt.figure.Figure(figsize=(10, 10))

    plot_confusion_matrix(cnf_matrix,
                          title="with-unknown-stomaco-" + modelname + "-comparison",classes=["predicted","Unknown"])