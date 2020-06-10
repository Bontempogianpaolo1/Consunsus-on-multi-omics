import pandas as pd
from Generation.VAE_model import VAE
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import torch
import numpy as np
import utils.custom_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
y = y.astype('category').cat.codes
seed = 1200
meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [mRNA_path,meth_path,mRNA_normalized_path]
filenames = [ "mrna","meth","mrna_normalized"]
for file, filename in zip(files, filenames):

    X =  pd.read_csv(file,index_col=False,header=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed,stratify=y)
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X_train)
    print("original silhoutte:")
    print(silhouette_score(X_train,y_train))
    for number_per_class in range(500,3000,500):
        X_train_new = pd.DataFrame(np.ndarray((0,X_train.shape[1])))
        y_train_new = pd.Series(np.ndarray(()))
        for label in range(5):
            #X_train_temp = X_train[y_train==label].copy()
            #y_train_temp = y_train[y_train==label].copy()

            dataset = utils.custom_dataset.CustomDataset(X_train_temp, y_train_temp.to_numpy(),
                                                         transform=utils.custom_dataset.ToTensor())
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
            model= VAE(X_train_temp.shape[1])
            #number_per_class=2000
            model.train_model(15,loader)
            sample = torch.randn(number_per_class-(y_train_temp==label).sum(), 100).to(model.device)
            sample = model.decode(sample).cpu()
            X_train_temp = pd.DataFrame(np.append(X_train_temp, sample.detach().numpy(), axis=0))
            y_train_temp = pd.Series(np.append(y_train_temp.values, np.ones((number_per_class - (y_train_temp == label).sum(),)) * label, axis=0))
            X_train_new=pd.DataFrame(np.append(X_train_new.to_numpy(),X_train_temp,axis=0))
            y_train_new=pd.Series(np.append(y_train_new.to_numpy(),y_train_temp.to_numpy(),axis=0))
        y_train_new=y_train_new.drop(index=0)
        print(filename+" silhoutte with "+str(number_per_class)+" samples per class:")
        print(silhouette_score(X_train_new, y_train_new))
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.scatter(X_train_new[y == 0, 0], X_train_new[y == 0, 1], X_train_new[y == 0, 2], c='r')
        ax.scatter(X_train_new[y == 1, 0], X_train_new[y == 1, 1], X_train_new[y == 1, 2], c='b')
        ax.scatter(X_train_new[y == 2, 0], X_train_new[y == 2, 1], X_train_new[y == 2, 2], c='g')
        ax.scatter(X_train_new[y == 3, 0], X_train_new[y == 3, 1], X_train_new[y == 3, 2], c='m')
        ax.scatter(X_train_new[y == 4, 0], X_train_new[y == 4, 1], X_train_new[y == 4, 2], c='y')
        print(filename)