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
from sklearn.metrics import classification_report


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
filenames = ["meth","mRNA", "miRNA"]
parameters = {'hidden_layer_sizes': np.arange(start=100, stop=150, step=10), 'random_state': [seed],
              'max_iter': [200, 400, 600]}
true_labels = []
for file, filename in zip(files, filenames):
    outputname = modelname + filename
    X = pd.read_csv(file, index_col=False, header=None)
    y2 = pd.read_csv("../Data/data/anomalies_preprocessed_annotation_global.csv")
    X2 = pd.read_csv("../Data/data/anomalies_preprocessed_Matrix_" + filename + ".csv", index_col=False, header=None)
    if filename == "miRNA":
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
        scores = np.empty([])
        components = np.empty([])
        variances = np.empty([])
        # clf=model
        if modelname == "bnn":
            clf = bnn.BNN(n_components, 20, 5)
            clf.train_step(X_train_transformed, y_train)
            tot, correct_predictions, predicted_for_images, new_prediction,probabilities = clf.test_batch(
                torch.from_numpy(X_test_transformed).float(), y_test, names, plot=False)
            y_pred = new_prediction
            maxprob = np.max(probabilities, axis=1)
            # maxprob= np.max(probabilities, axis=1)

        elif modelname == "mlp":
            clf = mlp.MLP(n_components, 20, 5)
            clf.train_step(X_train_transformed, y_train)
            probabilities, true_labels = clf.test_forced(X_test_transformed, y_test)
            y_pred = np.argmax(probabilities, axis=1)
            maxprob = np.max(probabilities, axis=1)
            y_pred[maxprob < 0.9] = 5

        elif modelname == "mlptree":
            clf = mlptree.MlpTree(n_components, 20, 5)
            clf.train_step(X_train_transformed, y_train)
            maxprob, y_pred, true_labels,probabilities = clf.test_forced(X_test_transformed, y_test)
            #maxprob = probabilities
            y_pred[maxprob < 0.9] = 5

        df = pd.DataFrame({
            'official_name': y_test['official_name'].tolist(),
            'max_probability': maxprob.tolist(),
            'probabilities': np.array(probabilities).tolist(),
            'y_pred': y_pred.tolist(),
            'y_true': y_test["label"].astype('category').cat.codes.tolist()
        })
        df.to_csv("../Data/outputs/pred-testset-" + modelname + "-" + filename + ".csv")
        outliers_names = y_test[y_pred == 5]['official_name']
        print(outliers_names)
        outliers_names.to_csv("../Data/outputs/outliers-testset-" + modelname + "-" + filename + ".csv", index=False)
        totalscore = accuracy_score(y_test['label'].astype('category').cat.codes, y_pred)
        # scores.append(totalscore)
        scores = np.append(scores, totalscore)
        components = np.append(components, n_components)
        ##components.append(n_components)
        # print("final score : %f" % totalscore)
        print("plot")
        cnf_matrix = confusion_matrix(y_test['label'].astype('category').cat.codes, y_pred)
        # plt.figure(figsize=(10, 10))
        # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
        print()
        np.set_printoptions(precision=2)
        # PlotDir non-normalized confusion matrix
        plt.figure.Figure(figsize=(10, 10))

        plot_confusion_matrix(cnf_matrix,
                              title="with-unknown-testset-"+modelname + "-" + filename,
                              classes=names2)
        with open("../Data/outputs/with-unknown-testset-"+modelname + "-" + filename+".txt", 'w') as f:
            print(classification_report(y_test['label'].astype('category').cat.codes, y_pred, ), file=f)

        if modelname == "bnn":
            # clf = bnn.BNN(n_components, 20, 5)
            # clf.train_step(X_train_transformed, y_train)
            tot, correct_predictions, predicted_for_images, new_prediction,probabilities = clf.test_batch(
                torch.from_numpy(X_transformed).float(), y2, names, plot=False)
            y_pred = new_prediction
            maxprob= np.max(probabilities, axis=1)

        elif modelname == "mlp":
            # clf = mlp.MLP(n_components, 20, 5)
            # clf.train_step(X_train_transformed, y_train)
            probabilities, true_labels = clf.test_forced(X_transformed, y2)
            y_pred = np.argmax(probabilities, axis=1)
            maxprob = np.max(probabilities, axis=1)
            y_pred[maxprob < 0.9] = 5

        elif modelname == "mlptree":
            # clf = mlptree.MlpTree(n_components, 20, 5)
            # clf.train_step(X_train_transformed, y_train)
            maxprob, y_pred, true_labels,probabilities = clf.test_forced(X_transformed, y2)
            #maxprob = probabilities
            y_pred[maxprob < 0.9] = 5
        df = pd.DataFrame({
            'official_name': y2['official_name'].tolist(),
            'max_probability': maxprob.tolist(),
            'probabilities': np.array(probabilities).tolist(),
            'y_pred': y_pred.tolist(),
            'y_true': y2["label"].astype('category').cat.codes.tolist()
        })
        df.to_csv("../Data/outputs/pred-stomaco-" + modelname + "-" + filename + ".csv")
        outliers_names = y2[y_pred == 5]['official_name']
        print(outliers_names)
        outliers_names.to_csv("../Data/outputs/outliers-stomaco-" + modelname + "-" + filename + ".csv",
                              index=False)
        print("plot")
        y_pred = y_pred.astype(np.float)
        y_true = y2['label'].astype('category').cat.codes
        y_true[y_true != 5] = 0
        y_true[y_true == 5] = 1
        y_pred[y_pred != 5] = 0
        y_pred[y_pred == 5] = 1

        cnf_matrix = confusion_matrix(y2['label'].astype('category').cat.codes, y_pred)
        # plt.figure(figsize=(10, 10))
        # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
        print()
        np.set_printoptions(precision=2)
        # PlotDir non-normalized confusion matrix
        plt.figure.Figure(figsize=(10, 10))

        plot_confusion_matrix(cnf_matrix,
                              title="with-unknown-stomaco-" + modelname + "-" + filename,
                              classes=["predicted","unknown"])

        with open("../Data/outputs/with-unknown-stomaco-"+modelname + "-" + filename+".txt", 'w') as f:
            print(classification_report(y2['label'].astype('category').cat.codes, y_pred, ), file=f)
