import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from utils.Plot import pareto_plot
import Classification.Mlp as mlp
import Classification.bnn as bnn
import Classification.Mlptree as mlptree
from matplotlib.ticker import PercentFormatter
seed = 1200
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)
#y = pd.read_csv(annotation_path)["label"]
#names = y.astype('category').cat.categories
#y = y.astype('category').cat.codes
modelname = " mlp "
meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path, mRNA_path, mRNA_normalized_path]
filenames = ["meth", "miRNA", "mRNA"]
parameters = {'hidden_layer_sizes': np.arange(start=100, stop=150, step=10), 'random_state': [seed],
              'max_iter': [200, 400, 600]}
true_labels = []
for file, filename in zip(files, filenames):
    outputname = modelname + filename
    X = pd.read_csv(file, index_col=False, header=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed,stratify=y["label"].astype('category').cat.codes)
    models = [bnn.BNN,mlp.MLP,mlptree.MlpTree]
    modelnames = ["bnn","mlp","mlptree"]
    pcamax = PCA()
    X_train_transformed_max = pcamax.fit_transform(X_train)
    X_test_transformed_max = pcamax.transform(X_test)
    var_max = X_test_transformed_max.var()
    print("real variance" + str(var_max))
    for model, modelname in zip(models, modelnames):
        scores = np.empty([])
        components = np.empty([])
        variances = np.empty([])
        #clf=model
        for n_components in range(10, 101, 10):
            pca = PCA(n_components=n_components)
            X_train_transformed = pca.fit_transform(X_train)
            Xtransformed = pca.fit_transform(X)
            X_test_transformed = pca.transform(X_test)
            variances = np.append(variances, X_test_transformed.var())
            if modelname=="bnn":
                clf = bnn.BNN(n_components, 20, 5)
                clf.train_step(X_train_transformed, y_train)
                probabilities, true_labels = clf.test_forced(X_test_transformed, y_test)
                y_pred = np.argmax(probabilities, axis=1)
            elif modelname=="mlp":
                clf = mlp.MLP(n_components, 20, 5)
                clf.train_step(X_train_transformed, y_train)
                probabilities, true_labels = clf.test_forced(X_test_transformed, y_test)
                y_pred = np.argmax(probabilities, axis=1)
            elif modelname=="mlptree":
                clf = mlptree.MlpTree(n_components, 20, 5)
                clf.train_step(X_train_transformed, y_train)
                probabilities, y_pred,true_labels = clf.test_forced(X_test_transformed, y_test)

            totalscore = accuracy_score(y_test['label'].astype('category').cat.codes, y_pred)
            # scores.append(totalscore)
            scores = np.append(scores, totalscore)
            components = np.append(components, n_components)
            ##components.append(n_components)
            print("variance:%f" % X_test_transformed.var())
            # print("final score : %f" % totalscore)
            del clf
        print("plot")
        # plot_pareto(var_max,variances[1:],components[1:],filename+" "+modelname)

        df = pd.DataFrame({
            'components': components[1:].tolist(),
            'variance': variances[1:].tolist(),
            'score': scores[1:].tolist(),
        })
        pareto_plot(df=df, title=filename + "pareto100" + modelname)
