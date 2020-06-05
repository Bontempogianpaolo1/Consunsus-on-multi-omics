import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
from pyro.distributions import Categorical, Normal
from pyro.infer import SVI, Trace_ELBO
import torch.optim as optim
import pandas as pd

import numpy as np
import utils.custom_dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import matplotlib as plt
from utils.Plot import plot_confusion_matrix


class MLP(nn.Module):

    def __init__(self, input_size=7, hidden_size=20, output_size=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.soft= nn.Softmax()

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return self.soft(output)

    def loss(self,y_pred,target):
        criterion= nn.CrossEntropyLoss()
        return criterion(y_pred,target.long())



num_iterations = 50
num_features=7



'''
for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(train_loader):
        # calculate the loss and take a gradient step
        loss += svi.step(data[0].view(-1, 28 * 28), data[1])
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train

    print("Epoch ", j, " Loss ", total_epoch_loss_train)

   '''
seed = 1200
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names = y.unique()
y = y.astype('category').cat.codes

meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path,mRNA_path,mRNA_normalized_path]
filenames = ["meth","mrna","micro mrna"]

for file, filename in zip(files, filenames):
    with open('../Data/outputs/'+filename+'-bnn-output.txt', 'w') as f:
        X = pd.read_csv(file,index_col=False,header=None)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
        pca = PCA(n_components=7)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        model = MLP()
        optimizer = optim.Adam(model.parameters())

        dataset = utils.custom_dataset.CustomDataset(X_train_transformed, y_train.to_numpy(),
                                                     transform=utils.custom_dataset.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        loss = 0
        for j in range(num_iterations):
            loss = 0
            for batch_id, data in enumerate(loader):
                # calculate the loss and take a gradient step
                pred=model(data["X"].view(-1, data["X"].shape[1]))
                loss = model.loss( pred,data["y"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            normalizer_train = len(loader.dataset)
            total_epoch_loss_train = loss.item() / normalizer_train

            print("Epoch ", j, " Loss ", total_epoch_loss_train)

        print(filename)

        num_samples = 10

        model.eval()
        print('Prediction when network is forced to predict')
        correct = 0
        total = 0
        dataset = utils.custom_dataset.CustomDataset(X_test_transformed, y_test.to_numpy(),
                                                     transform=utils.custom_dataset.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        probabilities = np.ndarray(shape=(0,5))
        true_labels= np.ndarray([])
        for j, data in enumerate(loader):
            images = data["X"]
            labels = data["y"]
            pred=model(data["X"].view(-1, data["X"].shape[1]))
            probabilities=np.append(probabilities,pred.detach().numpy(),axis=0)
            true_labels = np.append(true_labels, labels)
            total += labels.size(0)
            correct += (torch.from_numpy(np.argmax(pred.detach().numpy(), axis=1)) == labels).sum().item()
        print("accuracy: %d %%" % (100 * correct / total))
        import pandas as pd

        pd.DataFrame(probabilities).to_csv("../Data/outputs/pred-mlp-"+filename+".csv")
        pd.DataFrame(true_labels).to_csv("../Data/outputs/true-labels.csv")


