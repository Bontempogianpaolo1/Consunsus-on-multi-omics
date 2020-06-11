import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from Classification.Mlp import MLP
import utils.custom_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from utils.Plot import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from Classification.Plot_mlp_results import plot_mlp_results
import matplotlib as plt

class MlpTree(nn.Module):
    def __init__(self, input_size=7, hidden_size=20, output_size=5):
        super(MlpTree, self).__init__()
        self.root = MLP(input_size, hidden_size,2)
        self.ifcancer= MLP(input_size, hidden_size,3)
        self.notcancer=MLP(input_size, hidden_size,3)
        self.optimizer = optim.Adam(self.parameters())
        self.num_iterations = 50
        self.num_features = input_size

    def forward(self, x):
        out1 = self.root(x)
        cancer = np.argmax(out1,axis=1)
        x1=x[torch.argmax(out1,axis=1)==0,:]
        x2=x[torch.argmax(out1,axis=1)==1,:]
        out2=self.ifcancer(x1)
        out3= self.notcancer(x2)

        return out1,out2,out3

    def loss(self, out1,out2,out3, target1,target2):

        return self.root.loss(out1,target1)+self.ifcancer.loss(out2,target2)+self.notcancer.loss(out3,target2)

    def train_step(self, X, y_train, plot=False):
        self.train()
        y_train['is_tumor']=y_train['is_tumor'].astype('category').cat.codes
        y_train_tumor= y_train[y_train['is_tumor']==0]
        y_train_nottumor = y_train[y_train['is_tumor'] == 1]
        y_train_tumor['label'] = y_train_tumor['label'].astype('category').cat.codes
        y_train_nottumor['label'] = y_train_nottumor['label'].astype('category').cat.codes


        dataset = utils.custom_dataset.CustomDataset(X, y_train.to_numpy(),
                                                     transform=utils.custom_dataset.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        self.train()
        for j in range(self.num_iterations):
            loss = 0
            for batch_id, data in enumerate(loader):
                # calculate the loss and take a gradient step
                pred = self(data["X"].view(-1, data["X"].shape[1]))
                loss = self.loss(pred, data["y"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            normalizer_train = len(loader.dataset)
            total_epoch_loss_train = loss.item() / normalizer_train
            if plot:
                print("Epoch ", j, " Loss ", total_epoch_loss_train)

    def test_forced(self, X, y_test):
        self.eval()
        correct = 0
        total = 0
        dataset = utils.custom_dataset.CustomDataset(X, y_test.to_numpy(),
                                                     transform=utils.custom_dataset.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        probabilities = np.ndarray(shape=(0, 5))
        true_labels = np.ndarray([])
        for j, data in enumerate(loader):
            images = data["X"]
            labels = data["y"]
            pred = self(images.view(-1, images.shape[1]))
            probabilities = np.append(probabilities, pred.detach().numpy(), axis=0)
            true_labels = np.append(true_labels, labels)
            total += labels.size(0)
            correct += (torch.from_numpy(np.argmax(pred.detach().numpy(), axis=1)) == labels).sum().item()
        print("accuracy: %d %%" % (100 * correct / total))
        return probabilities, true_labels



num_iterations = 50
num_features = 7
seed = 1200
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)
#names = y.astype('category').cat.categories
#y = y.astype('category').cat.codes
meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path, mRNA_path, mRNA_normalized_path]
filenames = ["meth", "mrna", "micro mrna"]
modelname = "mlp"
for file, filename in zip(files, filenames):
    with open('../Data/outputs/' + filename + '-bnn-output.txt', 'w') as f:
        X = pd.read_csv(file, index_col=False, header=None)
        if filename == "mrna":
            X = pd.DataFrame(X[X.std().sort_values(ascending=False).head(1200).index].values.tolist())
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
        pca = PCA(n_components=num_features)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        clf = MlpTree(num_features, 20, 5)
        clf.train_step(X_train_transformed, y_train)
        probabilities, true_labels = clf.test_forced(X_test_transformed, y_test)
        print(filename)

        pd.DataFrame(probabilities).to_csv("../Data/outputs/pred-mlp-" + filename + ".csv")
        pd.DataFrame(true_labels).to_csv("../Data/outputs/true-labels.csv")


plot_mlp_results()


#if __name__ == "__main__":
#    print("Mlp imported")
