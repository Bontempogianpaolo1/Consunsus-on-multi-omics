import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from utils.Plot import plot_pareto,pareto_plot
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
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
filenames=["preprocessed-meth","preprocessed-mrna","preprocessed-micro mrna "]

parameters = { 'hidden_layer_sizes':np.arange(start=100,stop= 150,step=10), 'random_state':[seed],'max_iter': [200,400,600 ]}

true_labels=[]
for file,filename in zip(files,filenames):
    outputname=modelname+filename

    X= pd.read_csv(file,index_col=False,header=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=seed,stratify=y)
    # models = [MLPClassifier(),svm.SVC(),RandomForestClassifier()]
    models = [MLPClassifier()]
    # modelnames=["mlp","svm","Random Forest"]
    modelnames = ["mlp"]
    pcamax=PCA()
    X_train_transformed_max = pcamax.fit_transform(X_train)
    X_test_transformed_max = pcamax.transform(X_test)
    var_max=X_test_transformed_max.var()
    print("real variance"+str(var_max))
    for model,modelname in zip(models,modelnames):
        scores=np.empty([])
        components=np.empty([])
        variances=np.empty([])
        for n_components in range(7,8,1):
            pca = PCA(n_components=n_components)
            X_train_transformed= pca.fit_transform(X_train)
            Xtransformed = pca.fit_transform(X)
            X_test_transformed=pca.transform(X_test)
            variances=np.append(variances,X_test_transformed.var())
            df=pd.DataFrame(Xtransformed)
            df['label']=y
            title=filename+str(n_components)
            #df = pd.read_csv('https://raw.github.com/pandas-dev/pandas/master/pandas/tests/data/csv/iris.csv')
            print("pronto")
            pd.plotting.parallel_coordinates(
                df, 'label',
                color=('r', 'm', 'b','y','k'))
            plt.savefig("../Data/outputs/paretos/" + title + ".png")
            plt.show()

            #model= GridSearchCV(model,parameters)
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            totalscore = accuracy_score(y_test, y_pred)
            #scores.append(totalscore)
            scores = np.append(scores, totalscore)
            components = np.append(components, n_components)
            ##components.append(n_components)
            print("variance:%f"%X_test_transformed.var())
            #print("final score : %f" % totalscore)
        print("plot")
        #plot_pareto(var_max,variances[1:],components[1:],filename+" "+modelname)
        '''
        df = pd.DataFrame({
            'components': components[1:].tolist(),
            'variance': variances[1:].tolist(),
            'score': scores[1:].tolist(),
        })
        pareto_plot(df=df, x='components', y='score', title=filename + " score " + modelname)
        '''

annotation_path="../Data/data/annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names=y.astype('category').cat.categories
y=y.astype('category').cat.codes
modelname=" mlp "
meth_path="../Data/data/Matrix_meth.csv"
mRNA_path="../Data/data/Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path="../Data/data/Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"

files=[meth_path,mRNA_path,mRNA_normalized_path]
filenames=["meth","mrna","micro mrna "]

parameters = { 'hidden_layer_sizes':np.arange(start=100,stop= 150,step=10), 'random_state':[seed],'max_iter': [200,400,600 ]}

true_labels=[]

for file,filename in zip(files,filenames):
    outputname=modelname+filename
    X= pd.read_csv(file,index_col=False,header=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=seed,stratify=y)
    #models = [MLPClassifier(),svm.SVC(),RandomForestClassifier()]
    models = [MLPClassifier()]
    #modelnames=["mlp","svm","Random Forest"]
    modelnames = ["mlp"]
    pcamax=PCA()
    X_train_transformed_max = pcamax.fit_transform(X_train)
    X_test_transformed_max = pcamax.transform(X_test)
    var_max=X_test_transformed_max.var()
    print("real variance" + str(var_max))
    for model,modelname in zip(models,modelnames):
        scores=np.empty([])
        components=np.empty([])
        variances=np.empty([])
        for n_components in range(4,10,1):
            pca = PCA(n_components=n_components)
            X_train_transformed= pca.fit_transform(X)
            #X_test_transformed=pca.transform(X_test)
            variances=np.append(variances,X_test_transformed.var())
            #model= GridSearchCV(model,parameters)
            model.fit(X_train_transformed,y_train)
            y_pred=model.predict(X_test_transformed)
            plt.figure()

            pd.tools.plotting.parallel_coordinates(
                df[['mpg', 'displacement', 'cylinders', 'horsepower', 'weight', 'acceleration']],
                'mpg')

            plt.show()
            totalscore = accuracy_score(y_test, y_pred)
            #scores.append(totalscore)
            scores = np.append(scores, totalscore)
            components = np.append(components, n_components)
            #components.append(n_components)
            print("variance:%f"%X_test_transformed.var())
            #print("final score : %f" % totalscore)
        print("plot")
        #plot_pareto(var_max,variances[1:],components[1:],filename+" "+modelname)
        df = pd.DataFrame({
            'components': components[1:].tolist(),
            'variance': variances[1:].tolist(),
            'score': scores[1:].tolist(),
        })
        pareto_plot(df=df, x='components', y='score', title=filename+" "+modelname)