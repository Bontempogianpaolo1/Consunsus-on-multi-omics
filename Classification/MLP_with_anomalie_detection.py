import matplotlib as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data.dataset import Dataset

import utils.custom_dataset
from utils.Plot import plot_confusion_matrix
from utils.Plot import plot_outliers

num_iterations = 50
num_features = 3


class MLP(nn.Module):
    def __init__(self, input_size=num_features, hidden_size=20, output_size=5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.soft = nn.Softmax()

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return self.soft(output)

    def loss(self, y_pred, target):
        criterion = nn.CrossEntropyLoss()
        return criterion(y_pred, target.long())


seed = 1200
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names = y.astype('category').cat.categories
y = y.astype('category').cat.codes
meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path, mRNA_path, mRNA_normalized_path]
outliers = [LocalOutlierFactor(novelty=True), IsolationForest(), EllipticEnvelope(random_state=0), svm.OneClassSVM()]
filenames = ["meth", "mrna", "micro mrna"]
modelnames = ["mlp-local-outlier", "mlp-isolation-forest", "mlp-elliptic", "mlp-one-class"]
for modelname, outlier in zip(modelnames, outliers):
    for file, filename in zip(files, filenames):
        with open('../Data/outputs/' + filename + '-bnn-output.txt', 'w') as f:
            X = pd.read_csv(file, index_col=False, header=None)
            if (filename == "mrna"):
                X = pd.DataFrame(X[X.std().sort_values(ascending=False).head(1200).index].values.tolist())
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)
            outlier_detector = outlier
            pca = PCA(n_components=num_features)
            X_train_transformed = pca.fit_transform(X_train)
            X_test_transformed = pca.transform(X_test)
            model = MLP()
            outlier_detector.fit(X_train_transformed)
            Not_Outliers = outlier_detector.predict(X_train_transformed)
            X_train_transformed = X_train_transformed[Not_Outliers == 1]
            y_train = y_train[Not_Outliers == 1]
            Not_Outliers = outlier_detector.predict(X_test_transformed)
            plot_outliers(X_test_transformed, Not_Outliers, X_train_transformed, modelname + filename + "pca")
            X_test_transformed = X_test_transformed[Not_Outliers == 1]
            y_test = y_test[Not_Outliers == 1]
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
                    pred = model(data["X"].view(-1, data["X"].shape[1]))
                    loss = model.loss(pred, data["y"])
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
            probabilities = np.ndarray(shape=(0, 5))
            true_labels = np.ndarray([])
            for j, data in enumerate(loader):
                images = data["X"]
                labels = data["y"]
                pred = model(data["X"].view(-1, data["X"].shape[1]))
                probabilities = np.append(probabilities, pred.detach().numpy(), axis=0)
                true_labels = np.append(true_labels, labels)
                total += labels.size(0)
                correct += (torch.from_numpy(np.argmax(pred.detach().numpy(), axis=1)) == labels).sum().item()
            print("accuracy: %d %%" % (100 * correct / total))
            import pandas as pd

            pd.DataFrame(probabilities).to_csv("../Data/outputs/pred-" + modelname + filename + ".csv")
            pd.DataFrame(true_labels).to_csv("../Data/outputs/true-labels.csv")
            cnf_matrix = confusion_matrix(true_labels[1:], np.argmax(probabilities, axis=1))
            print()
            np.set_printoptions(precision=2)
            # PlotDir non-normalized confusion matrix
            plt.figure.Figure(figsize=(10, 10))

            plot_confusion_matrix(cnf_matrix, title=modelname + filename, classes=names)

            X2 = pd.read_csv("../Data/data/anomalies_preprocessed_Matrix_" + filename + ".csv", index_col=False,
                             header=None)
            y2 = pd.read_csv("../Data/data/anomalies_preprocessed_annotation_global.csv")["label"]
            if filename == "mrna":
                X2 = pd.DataFrame(X2[X2.std().sort_values(ascending=False).head(1200).index].values.tolist())
            X_transformed = pca.transform(X2)
            y_pred = outlier_detector.predict(X_transformed)
            plot_outliers(X_transformed, y_pred, X_train_transformed, "newdata-" + modelname + "-" + filename + "pca")
            y_pred[y_pred == -1] = 0
            true_label = y2.astype('category').cat.codes.to_numpy().copy()
            true_label[true_label == 0] = 1
            cnf_matrix = confusion_matrix(true_label, y_pred)

            print()
            np.set_printoptions(precision=2)
            # PlotDir non-normalized confusion matrix
            plt.figure.Figure(figsize=(10, 10))
            print("tot predicted")
            print((y_pred == 1).sum())
            plot_confusion_matrix(cnf_matrix,
                                  title=modelname + "-anomalies-" + filename,
                                  classes=["predicted", "Unknown"])
