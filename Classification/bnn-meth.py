import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyro
from pyro.distributions import Categorical,Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
import numpy as np
import utils.custom_dataset
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import matplotlib as plt
from utils.Plot import plot_confusion_matrix

class NN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output

net = NN(10000, 1024, 5)
log_softmax = nn.LogSoftmax(dim=1)

def model(x_data, y_data):
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    lhat = log_softmax(lifted_reg_model(x_data))
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


num_samples = 10
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)


softplus = torch.nn.Softplus()


def guide(x_data, y_data):
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    lifted_module = pyro.random_module("module", net, priors)
    return lifted_module()


optim = Adam({"lr": 0.01})

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
seed=1200
annotation_path="../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names=y.unique()
y=y.astype('category').cat.codes


meth_path="../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path="../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path="../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files=[meth_path]
filenames=["meth"]
for file,filename in zip(files,filenames):

    X= pd.read_csv(file).drop(columns=["Composite Element REF","Unnamed: 0"])
    X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=seed)
    dataset= utils.custom_dataset.CustomDataset(X_train.to_numpy(),y_train.to_numpy(),transform= utils.custom_dataset.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    num_iterations = 30
    loss = 0
    for j in range(num_iterations):
        loss = 0
        for batch_id, data in enumerate(loader):
            # calculate the loss and take a gradient step
            loss += svi.step(data["X"].view(-1, data["X"].shape[1]), data["y"])
        normalizer_train = len(loader.dataset)
        total_epoch_loss_train = loss / normalizer_train

        print("Epoch ", j, " Loss ", total_epoch_loss_train)


    print(filename)

    num_samples = 10
    def predict(x):
        sampled_models = [guide(None, None) for _ in range(num_samples)]
        yhats = [model(x).data for model in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return np.argmax(mean.numpy(), axis=1)

    print('Prediction when network is forced to predict')
    correct = 0
    total = 0
    dataset = utils.custom_dataset.CustomDataset(X_test.to_numpy(), y_test.to_numpy(),transform= utils.custom_dataset.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for j, data in enumerate(loader):
        images= data["X"]
        labels=data["y"]
        predicted = predict(images.view(-1,10000))
        total += labels.size(0)
        correct += (torch.from_numpy(predicted) == labels).sum().item()
    print("accuracy: %d %%" % (100 * correct / total))


