from itertools import cycle, islice

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
import itertools
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def plot_roc(n_classes, y_score, y_test2, names, title):
    y_score = pd.factorize(y_score, )
    y_test2 = pd.factorize(y_test2, sort=True)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes - 1):
        fpr[i], tpr[i], _ = roc_curve(y_test2[0], y_score[0], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area

    # Finally average it and compute AUC

    lw = 2
    # PlotDir all ROC curve
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    for i, color in zip(range(n_classes - 1), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.pause(0.2)

def plot_pareto(max_variance,variance,n_components,title):
    fig, ax = plt.subplots()
    max_variance=variance.max()
    #variance=variance/max_variance*100
    ax.bar(n_components, variance, color="C0")
    ax2 = ax.twinx()
    ax2.plot(n_components, variance.cumsum()/variance.sum(), color="C1", marker="D")
    ax2.yaxis.set_major_formatter(PercentFormatter())

    ax.tick_params(axis="y", colors="C0")
    ax.set_ylabel('variance(variance / best) ')
    ax.set_xlabel('number of principal components')
    ax2.tick_params(axis="y", colors="C1")
    ax2.set_ylabel('variance cumulative')
    plt.title(title)
    plt.savefig("../Data/outputs/paretos/" + title + ".png")
    plt.show()


def pareto_plot(df, title=None, show_pct_y=False, pct_format='{:.1%}'):
    """
           df = pd.DataFrame({
               'components': components[1:].tolist(),
               'variance': variances[1:].tolist(),
               'score': scores[1:].tolist(),
           })
    """
    tmp = df
    x = tmp['components'].values
    variances = tmp['variance'].values
    scores = tmp['score'].values
    weights = variances / variances.sum()
    cumsum = weights.cumsum()

    fig, ax1 = plt.subplots()
    ax1.bar(x, variances)
    ax1.set_xlabel("pc")
    ax1.set_ylabel("variance")

    ax2 = ax1.twinx()
    ax2.plot(x, scores, '-ro', alpha=0.5)
    ax2.set_ylabel('accuracy', color='r')
    ax2.tick_params('y', colors='r')
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    #ax2.yaxis.set_major_formatter(PercentFormatter())
    # hide y-labels on right side
    #if not show_pct_y:
    #    ax2.set_yticks([])
    formatted_weights = [pct_format.format(x) for x in scores]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], scores[i]), fontweight='heavy')

    ax3 = ax1.twinx()
    offset=60
    ax3.spines["right"].set_position(("axes", 1.2))

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    make_patch_spines_invisible(ax3)
    # Second, show the right spine.
    ax3.spines["right"].set_visible(True)
    ax3.plot(x, cumsum, '-yo', alpha=0.5)
    ax3.set_ylabel('cumulative variance', color='y')
    ax3.tick_params('y', colors='y')
    vals = ax3.get_yticks()
    ax3.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    # hide y-labels on right side
    #ax3.yaxis.set_major_formatter(PercentFormatter())
    #if not show_pct_y:
    #    ax3.set_yticks([])
    formatted_weights = [pct_format.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax3.annotate(txt, (x[i], cumsum[i]), fontweight='heavy')

    #if title:
    #    plt.title(title)

    plt.tight_layout()
    plt.savefig("../Data/outputs/paretos/" + title + ".png")
    plt.show()
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("../Data/outputs/"+title + ".png")
    plt.pause(0.2)


# Learning curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.pause(0.2)


def plot_diagram(df):
    # number of elements per class
    counts = df.iloc[:, 0].value_counts()
    # different classes
    names = counts.index
    # tot classes
    x = np.arange(counts.shape[0])
    plt.close()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
    plt.title("Elements per class")
    colors = np.array(list(islice(cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k']), int(counts.shape[0] + 1))))
    ax1.barh(x, counts, 0.75, color=colors)
    ax1.set_yticks(x + 0.75 / 2)
    ax1.set_xticks([])
    ax1.set_yticklabels(names, minor=False)
    wedges, texts, autotexts = ax2.pie(counts, autopct=lambda pct: "{:.1f}%\n".format(pct))
    ax2.legend(wedges, names, title="Ingredients", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    for i, v in enumerate(counts):
        ax1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    plt.pause(0.2)
    print(counts)
    print("diagram displayed")
    print("===================")



def plot_outliers(X,y,X_train,title):
    #pca=PCA(n_components=3)
    #X_train=pca.fit_transform(X_train)
    #X=pca.transform(X)
    #X=pca.transform(X)
    #X_train=pca.transform(X_train)
    #X_train=X_train.to_numpy()
    colors = np.array(list(islice(cycle(['b', 'r', 'g', 'c', 'm', 'y', 'k']), 4)))
    markers = np.array(list(islice(cycle(['o', 'v', '^', '<', '>', '8', 's']), 4)))
    lista = [1,-1]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    for l, c, m in zip(lista, colors, markers):
        ax.scatter(X[y == l, 0], X[y == l, 1], X[y == l, 2], c=c, marker=m,label ='class %s' % l)

    colors = np.array(list(islice(cycle([ 'g', 'c', 'm', 'y', 'k','b', 'r']), 4)))
    markers = np.array(list(islice(cycle(['o', 'v', '^', '<', '>', '8', 's']), 4)))
    lista = [1]
    for l, c, m in zip(lista, colors, markers):
        ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c=c, marker=m,label ='train-set without outliers')
    ax.legend(loc='upper left', fontsize=12)
    plt.title(title)
    plt.savefig("../Data/outputs/" + title + ".png")
    plt.pause(0.2)
