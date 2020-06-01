from torch import nn,optim
import torch
import torch.functional as F

class VAE(nn.Module):
    def __init__(self,size):
        super(VAE, self).__init__()
        self.size=size
        self.device = torch.device("cpu")
        self.fc1 = nn.Linear(size, 1000)
        self.fc21 = nn.Linear(1000, 100)
        self.fc22 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1000)
        self.fc4 = nn.Linear(1000, size)
        self.r1=nn.ReLU()
        self.r2=nn.ReLU()

    def encode(self, x):
        h1 = self.r1(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.r2(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,recon_x, x, mu, logvar):
        criterion = nn.MSELoss()
        BCE = 0
        BCE += criterion(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train_model(self,epochs,train_loader):
        self.train()
        train_loss = 0
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(epochs):
            for batch_idx, data in enumerate(train_loader):
                data=data['X']
                data = data.to(self.device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                '''                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data)))'''


        #print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


    def test(self,epoch,test_loader):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()



        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))