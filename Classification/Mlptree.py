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
        self.leaf1= MLP(input_size, hidden_size,output_size)
        self.leaf2=MLP(input_size, hidden_size,output_size)
        self.optimizer = optim.Adam(self.parameters())
        self.num_iterations = 50
        self.num_features = input_size

    def forward(self, x):
        out1 = self.root(x)
        prob0=torch.max(out1,dim=1).values
        predictions=prob0.data.new(prob0.size()).long()
        cancer = torch.argmax(out1,axis=1)
        final_probabilities=prob0.new_empty((prob0.size(0),5))
        if(cancer==0).sum()>0:
            x1 = x[cancer == 0, :]
            out2 = self.leaf1(x1)
            pred1=torch.argmax(out2,axis=1)
            prob1 = torch.max(out2,dim=1).values
            final_pred1=prob1*prob0[cancer==0]

            final_probabilities[cancer==0] = (out2*prob0.view(x.shape[0],1)[cancer==0,:])
            prob0[cancer == 0] = final_pred1
            predictions[cancer==0]= pred1
        if (cancer == 1).sum() > 0:
            x2 = x[cancer == 1, :]
            out3 = self.leaf2(x2)
            pred2=torch.argmax(out3,axis=1)
            prob2=torch.max(out3,dim=1).values
            final_pred2 = prob2 * prob0[cancer == 1]
            final_probabilities[cancer == 1] =(out3*prob0.view(x.shape[0],1)[cancer==1,:])
            prob0[cancer == 1] = final_pred2
            predictions[cancer == 1] = pred2

        return predictions,prob0,final_probabilities

    def loss(self, out1,out2,out3, target1,target2):

        return self.root.loss(out1,target1)+self.ifcancer.loss(out2,target2)+self.notcancer.loss(out3,target2)

    def train_step(self, X, y_train, plot=False):
        self.train()
        y_train['is_tumor']=y_train['is_tumor'].astype('category').cat.codes
        y_train['label'] = y_train['label'].astype('category').cat.codes
        X_tumor =X[y_train['is_tumor']==0]
        X_nottumor=X[y_train['is_tumor'] == 1]
        y_train_tumor= y_train[y_train['is_tumor']==0]
        y_train_nottumor = y_train[y_train['is_tumor'] == 1]
        self.root.train_step2(X,y_train['is_tumor'],True)
        self.leaf1.train_step2(X_tumor,y_train_tumor['label'],True)
        self.leaf2.train_step2(X_nottumor, y_train_nottumor['label'],True)


    def test_forced(self, X, y_test):
        self.eval()
        correct = 0
        total = 0
        dataset = utils.custom_dataset.CustomDataset2(X, y_test['label'].astype('category').cat.codes.to_numpy(),y_test['official_name'].to_numpy(),
                                                     transform=utils.custom_dataset.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        probabilities = np.ndarray(shape=(0, ))
        full_probs = np.ndarray(shape=(0,5 ))
        predictions = np.ndarray(shape=(0,))
        true_labels = np.ndarray([])
        for j, data in enumerate(loader):
            images = data["X"]
            labels = data["y"]
            prediction,maxprob,fprob = self(images.view(-1, images.shape[1]))
            probabilities = np.append(probabilities, maxprob.detach().numpy(), axis=0)
            full_probs = np.append(full_probs,fprob.detach().numpy(), axis=0)
            predictions = np.append(predictions, prediction.detach().numpy(), axis=0)
            true_labels = np.append(true_labels, labels)
            total += len(labels)
            correct += (prediction == labels).sum().item()
        print("accuracy: %d %%" % (100 * correct / total))
        return probabilities, predictions,true_labels,full_probs

'''

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

'''
if __name__ == "__main__":
    print("Mlp imported")
