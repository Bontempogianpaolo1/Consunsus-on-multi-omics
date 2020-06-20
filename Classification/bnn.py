import numpy as np
import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import Categorical, Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data.dataset import Dataset

import utils.custom_dataset


class BNN(nn.Module):
    def __init__(self, input_size=7, hidden_size=20, output_size=5):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softplus = torch.nn.Softplus()
        self.optim = Adam({"lr": 0.01})
        self.num_features = input_size
        self.num_iterations = 50
        self.num_samples=100

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output

    def model(self, x_data, y_data):
        fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias))
        outw_prior = Normal(loc=torch.zeros_like(self.out.weight), scale=torch.ones_like(self.out.weight))
        outb_prior = Normal(loc=torch.zeros_like(self.out.bias), scale=torch.ones_like(self.out.bias))
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()
        lhat = self.log_softmax(lifted_reg_model(x_data))
        pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

    def predict(self, x):
        num_samples = 10
        sampled_models = [self.guide(None, None) for _ in range(num_samples)]
        yhats = [model(x).data for model in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return np.argmax(mean.numpy(), axis=1), mean

    def guide(self, x_data, y_data):
        # First layer weight distribution priors
        fc1w_mu = torch.randn_like(self.fc1.weight)
        fc1w_sigma = torch.randn_like(self.fc1.weight)
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = self.softplus(pyro.param("fc1w_sigma", fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        # First layer bias distribution priors
        fc1b_mu = torch.randn_like(self.fc1.bias)
        fc1b_sigma = torch.randn_like(self.fc1.bias)
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = self.softplus(pyro.param("fc1b_sigma", fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
        # Output layer weight distribution priors
        outw_mu = torch.randn_like(self.out.weight)
        outw_sigma = torch.randn_like(self.out.weight)
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = self.softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
        # Output layer bias distribution priors
        outb_mu = torch.randn_like(self.out.bias)
        outb_sigma = torch.randn_like(self.out.bias)
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = self.softplus(pyro.param("outb_sigma", outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
        lifted_module = pyro.random_module("module", self, priors)
        return lifted_module()

    def train_step(self, X, y_train, plot=False):
        self.train()
        y_train = y_train["label"].astype('category').cat.codes
        pyro.clear_param_store()
        dataset = utils.custom_dataset.CustomDataset(X, y_train.to_numpy(),
                                                     transform=utils.custom_dataset.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        svi = SVI(self.model, self.guide, self.optim, loss=Trace_ELBO())
        for j in range(self.num_iterations):
            loss = 0
            for batch_id, data in enumerate(loader):
                # calculate the loss and take a gradient step
                loss += svi.step(data["X"].view(-1, data["X"].shape[1]), data["y"])
            normalizer_train = len(loader.dataset)
            total_epoch_loss_train = loss / normalizer_train
            if plot:
                print("Epoch ", j, " Loss ", total_epoch_loss_train)

    def test_forced(self, X, y_test):
        self.eval()
        y_test = y_test["label"].astype('category').cat.codes
        dataset = utils.custom_dataset.CustomDataset(X, y_test.to_numpy(),
                                                     transform=utils.custom_dataset.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        probabilities = np.ndarray(shape=(0, 5))
        true_labels = np.ndarray([])
        correct = 0
        total = 0
        for j, data in enumerate(loader):
            images = data["X"]
            labels = data["y"]
            predicted, mean = self.predict(images.view(-1, self.num_features))
            probabilities = np.append(probabilities, mean, axis=0)
            true_labels = np.append(true_labels, labels)
            total += labels.size(0)
            correct += (torch.from_numpy(predicted) == labels).sum().item()
        print("accuracy: %d %%" % (100 * correct / total))
        return probabilities, true_labels

    def give_uncertainities(self, x):
        sampled_models = [self.guide(None, None) for _ in range(self.num_samples)]
        yhats = [F.log_softmax(model(x.view(-1, self.num_features)).data, 1).detach().numpy() for model in
                 sampled_models]
        return np.asarray(yhats)

    def test_batch(self, images, labels, classes, plot=True):
        labels=labels["label"].astype('category').cat.codes.to_numpy()
        y = self.give_uncertainities(images)
        predicted_for_images = 0
        correct_predictions = 0
        new_prediction = labels.copy()
        probabilities = []
        for i in range(len(labels)):
            all_digits_prob = []
            highted_something = False
            temp_probabilities=[]
            for j in range(len(classes)):
                highlight = False
                histo = []
                histo_exp = []
                for z in range(y.shape[0]):
                    histo.append(y[z][i][j])
                    histo_exp.append(np.exp(y[z][i][j]))

                prob = np.percentile(histo_exp, 50)  # sampling median probability
                temp_probabilities.append(prob)
                if prob > 0.9:  # select if network thinks this sample is 20% chance of this being a label
                    highlight = True  # possibly an answer
                all_digits_prob.append(prob)
                if highlight:
                    highted_something = True
            predicted = np.argmax(all_digits_prob)
            probabilities.append(temp_probabilities)
            if highted_something:
                new_prediction[i] = predicted
                predicted_for_images += 1
                if labels[i].item() == predicted:
                    correct_predictions += 1.0
            else:
                new_prediction[i] = 5
        return len(labels), correct_predictions, predicted_for_images, new_prediction,probabilities


'''
seed = 1200
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names = y.unique()
y = y.astype('category').cat.codes
meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path, mRNA_path, mRNA_normalized_path]
filenames = ["meth", "mrna", "micro mrna"]

for file, filename in zip(files, filenames):
    with open('../Data/outputs/' + filename + '-bnn-output.txt', 'w') as f:
        X = pd.read_csv(file, index_col=False, header=None)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)
        pca = PCA(n_components=num_features)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        clf = BNN(num_features, 20, 5)
        clf.train_step(X_train_transformed, y_train)
        print('Prediction when network is forced to predict')
        probabilities, true_labels = clf.test_forced(X_test_transformed, y_test)

        import pandas as pd

        pd.DataFrame(probabilities).to_csv("../Data/outputs/pred-bnn-" + filename + ".csv")
        pd.DataFrame(true_labels).to_csv("../Data/outputs/true-labels.csv")
plot_bnn_results()
'''
if __name__ == "__main__":
    print("bnn imported")
