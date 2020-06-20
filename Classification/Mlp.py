import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import pandas as pd
import utils.custom_dataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from Classification.Plot_mlp_results import plot_mlp_results
import matplotlib as plt
from utils.Plot import plot_confusion_matrix
class MLP(nn.Module):
    def __init__(self, input_size=7, hidden_size=20, output_size=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softmax()
        self.optimizer = optim.Adam(self.parameters())
        self.num_iterations = 50
        self.num_features = input_size

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return self.soft(output)

    def loss(self, y_pred, target):
        criterion = nn.CrossEntropyLoss()
        return criterion(y_pred, target.long())

    def train_step(self, X, y_train, plot=False):
        self.train()
        y_train = y_train["label"].astype('category').cat.codes
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

    def train_step2(self, X, y_train, plot=False):
        self.train()
        #y_train = y_train["label"].astype('category').cat.codes
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
        y_test = y_test["label"].astype('category').cat.codes
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






if __name__ == "__main__":
    print("Mlp imported")
    num_iterations = 50
    num_features = 7
    seed = 1200
    annotation_path = "../Data/data/preprocessed_annotation_global.csv"
    y = pd.read_csv(annotation_path)
    names = y["label"].astype('category').cat.categories
    #y = y.astype('category').cat.codes
    meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
    mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
    mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
    files = [meth_path, mRNA_path, mRNA_normalized_path]
    filenames = ["meth", "miRNA", "mRNA"]
    modelname = "mlp"
    for file, filename in zip(files, filenames):
        with open('../Data/outputs/' + filename + '-bnn-output.txt', 'w') as f:
            X = pd.read_csv(file, index_col=False, header=None)
            if filename == "mrna":
                X = pd.DataFrame(X[X.std().sort_values(ascending=False).head(1200).index].values.tolist())
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y["label"])
            pca = PCA(n_components=num_features)
            X_train_transformed = pca.fit_transform(X_train)
            X_test_transformed = pca.transform(X_test)
            clf = MLP(num_features, 20, 5)
            clf.train_step(X_train_transformed, y_train)
            probabilities, true_labels = clf.test_forced(X_test_transformed, y_test)
            print(filename)

            pd.DataFrame(probabilities).to_csv("../Data/outputs/pred-mlp-" + filename + ".csv")
            pd.DataFrame(true_labels).to_csv("../Data/outputs/true-labels.csv")

            X2 = pd.read_csv("../Data/data/anomalies_preprocessed_Matrix_" + filename + ".csv", index_col=False,
                             header=None)
            y2 = pd.read_csv("../Data/data/anomalies_preprocessed_annotation_global.csv")["label"]
            if filename == "mrna":
                X2 = pd.DataFrame(X2[X2.std().sort_values(ascending=False).head(1200).index].values.tolist())
            X_transformed = pca.transform(X2)
            print('Prediction when network is forced to predict')

            probabilities, true_labels = clf.test_forced(X_transformed, y2.astype('category').cat.codes)
            y_pred=  np.argmax(probabilities, axis=1)
            y_pred[np.max(probabilities,axis=1)<0.6]=5
            cnf_matrix = confusion_matrix(true_labels[1:], np.argmax(probabilities, axis=1))
            # plt.figure(figsize=(10, 10))
            # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
            print()
            np.set_printoptions(precision=2)
            # PlotDir non-normalized confusion matrix
            plt.figure.Figure(figsize=(10, 10))

            plot_confusion_matrix(cnf_matrix,
                                  title=modelname + "-anomalies-" + filename,
                                  classes=names)
    plot_mlp_results()
