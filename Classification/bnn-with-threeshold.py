import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyro
from pyro.distributions import Categorical, Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
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
num_iterations = 5
num_features=7
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


net = NN(num_features, 20, 5)
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
    return np.argmax(mean.numpy(), axis=1),mean


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


def give_uncertainities(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [F.log_softmax(model(x.view(-1,num_features)).data, 1).detach().numpy() for model in sampled_models]
    return np.asarray(yhats)


def test_batch(images, labels,classes, plot=True):
    y = give_uncertainities(images)
    predicted_for_images = 0
    correct_predictions = 0
    new_prediction=labels.copy()
    for i in range(len(labels)):

        if (plot):
            print("Real: ", labels[i].item())
            fig, axs = plt.subplots(1, 5, sharey=True, figsize=(20, 2))

        all_digits_prob = []

        highted_something = False

        for j in range(len(classes)):

            highlight = False

            histo = []
            histo_exp = []

            for z in range(y.shape[0]):
                histo.append(y[z][i][j])
                histo_exp.append(np.exp(y[z][i][j]))

            prob = np.percentile(histo_exp, 50)  # sampling median probability
            print(prob)
            if (prob > 0.5):  # select if network thinks this sample is 20% chance of this being a label
                highlight = True  # possibly an answer

            all_digits_prob.append(prob)
            if (plot):
                N, bins, patches = axs[j].hist(histo, bins=8, color="lightgray", lw=0,
                                               weights=np.ones(len(histo)) / len(histo), density=False)
                axs[j].set_title(str(classes[j]) + " (" + str(round(prob, 2)) + ")")

            if (highlight):

                highted_something = True

                if (plot):

                    # We'll color code by height, but you could use any scalar
                    fracs = N / N.max()

                    # we need to normalize the data to 0..1 for the full range of the colormap
                    norm = colors.Normalize(fracs.min(), fracs.max())

                    # Now, we'll loop through our objects and set the color of each accordingly
                    for thisfrac, thispatch in zip(fracs, patches):
                        color = plt.cm.viridis(norm(thisfrac))
                        thispatch.set_facecolor(color)

        if (plot):
            plt.show()

        predicted = np.argmax(all_digits_prob)
        print(i)
        if (highted_something):
            new_prediction[i]=predicted
            predicted_for_images += 1
            if (labels[i].item() == predicted):
                if (plot):
                    print("Correct")
                correct_predictions += 1.0
            else:
                if (plot):
                    print("Incorrect :()")
        else:
            new_prediction[i] = 5
            if (plot):
                print("Undecided.")

    if (plot):
        print("Summary")
        print("Total images: ", len(labels))
        print("Predicted for: ", predicted_for_images)
        print("Accuracy when predicted: ", correct_predictions / predicted_for_images)
    totalscore = accuracy_score(labels, new_prediction)
    print("final score : %f" % totalscore)
    return len(labels), correct_predictions, predicted_for_images,new_prediction


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
seed = 1200
annotation_path = "../Data/data/preprocessed_annotation_global.csv"
y = pd.read_csv(annotation_path)["label"]
names = y.astype('category').cat.categories

y = y.astype('category').cat.codes

meth_path = "../Data/data/preprocessed_Matrix_meth.csv"
mRNA_path = "../Data/data/preprocessed_Matrix_miRNA_deseq_correct.csv"
mRNA_normalized_path = "../Data/data/preprocessed_Matrix_mRNA_deseq_normalized_prot_coding_correct.csv"
files = [meth_path,mRNA_path,mRNA_normalized_path]
filenames = ["meth","mrna","micro mrna"]
modelname="bnn-with-threeshold"
for file, filename in zip(files, filenames):
    with open('../Data/outputs/'+filename+'-'+modelname+'-output.txt', 'w') as f:
        X = pd.read_csv(file,index_col=False,header=None)
        if (filename == "mrna"):
            X = pd.DataFrame(X[X.std().sort_values(ascending=False).head(1200).index].values.tolist())
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
        pca = PCA(n_components=7)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        dataset = utils.custom_dataset.CustomDataset(X_train_transformed, y_train.to_numpy(),
                                                     transform=utils.custom_dataset.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        svi = SVI(model, guide, optim, loss=Trace_ELBO())

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
            return np.argmax(mean.numpy(), axis=1),mean

        '''
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
            predicted,mean = predict(images.view(-1, num_features))
            probabilities=np.append(probabilities,mean,axis=0)
            true_labels = np.append(true_labels, labels)
            total += labels.size(0)
            correct += (torch.from_numpy(predicted) == labels).sum().item()
        print("accuracy: %d %%" % (100 * correct / total))
        import pandas as pd

        pd.DataFrame(probabilities).to_csv("../Data/outputs/pred-bnn-"+filename+".csv")
        pd.DataFrame(true_labels).to_csv("../Data/outputs/true-labels.csv")
        '''
        tot, correct_predictions, predicted_for_images,new_prediction=test_batch(torch.from_numpy(X_test_transformed).float(),y_test.to_numpy(),names,plot=False)
        print("Summary")
        print("Total images: ", tot)
        print("Predicted for: ", predicted_for_images)
        print("Accuracy when predicted: ", correct_predictions / predicted_for_images)
        print("Confusion matrix")
        #totalscore = accuracy_score(y_test,new_prediction)
        #print("final score : %f" % totalscore)
        cnf_matrix = confusion_matrix(y_test, new_prediction)
        # plt.figure(figsize=(10, 10))
        # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
        print()
        np.set_printoptions(precision=2)
        # PlotDir non-normalized confusion matrix
        plt.figure.Figure(figsize=(10, 10))

        plot_confusion_matrix(cnf_matrix,
                              title=modelname + "-" + filename, classes=names.append(pd.Index(["Unknown"])))

        #print(modelname + filename + " " + str(model.best_params_), file=f)
        print(classification_report(y_test, new_prediction, ), file=f)
        X2 = pd.read_csv("../Data/data/anomalies_preprocessed_Matrix_" + filename + ".csv", index_col=False, header=None)
        y2 = pd.read_csv("../Data/data/anomalies_preprocessed_annotation_global.csv")["label"]
        if(filename=="mrna"):
            X2 = pd.DataFrame(X2[X2.std().sort_values(ascending=False).head(1200).index].values.tolist())
        X_transformed = pca.transform(X2)

        tot, correct_predictions, predicted_for_images, new_prediction = test_batch(
            torch.from_numpy(X_transformed).float(), y2.astype('category').cat.codes.to_numpy(),
            y2.astype('category').cat.categories, plot=False)
        print("Summary")
        print("Total images: ", tot)
        print("Predicted for: ", predicted_for_images)
        #print("Accuracy when predicted: ", correct_predictions / predicted_for_images)
        print("Confusion matrix")
        # totalscore = accuracy_score(y_test,new_prediction)
        # print("final score : %f" % totalscore)
        cnf_matrix = confusion_matrix(y2.astype('category').cat.codes.to_numpy(), new_prediction)
        # plt.figure(figsize=(10, 10))
        # plot_roc(names.shape[0], y_pred, y_test_bal, names, title)
        print()
        np.set_printoptions(precision=2)
        # PlotDir non-normalized confusion matrix
        plt.figure.Figure(figsize=(10, 10))

        plot_confusion_matrix(cnf_matrix,
                              title=modelname + "-anomalies-" + filename,
                              classes=y2.astype('category').cat.categories.append(pd.Index(["Unknown"])))

        # print(modelname + filename + " " + str(model.best_params_), file=f)
        print(classification_report(y2.astype('category').cat.codes.to_numpy(), new_prediction, ), file=f)