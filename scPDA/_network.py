import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        layer1=100,
        layer2=50,
        n_hidden=15,
        alpha_init=None,
        theta1_init=None,
        theta2_init=None,
    ):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.means = nn.Linear(layer2, n_hidden)
        self.logvars = nn.Linear(layer2, n_hidden)

        # Decoder
        self.fc3 = nn.Linear(n_hidden, layer2)
        self.fc4 = nn.Linear(layer2, layer1)
        self.pi = nn.Linear(layer1, input_dim)

        # Global Parameters
        self.alpha = nn.Parameter(alpha_init.clone().detach())
        self.theta1 = nn.Parameter(theta1_init.clone().detach())
        self.theta2 = nn.Parameter(theta2_init.clone().detach())

    def reparametrize(self, means, logvars):
        var = logvars.exp() + 1e-5
        z = Normal(means, var.sqrt()).rsample()
        return z

    def encoder(self, x):
        enc = torch.log1p(x)
        enc = self.fc1(enc)
        enc = F.selu(enc)
        enc = self.fc2(enc)
        enc = F.selu(enc)

        means = self.means(enc)
        logvars = self.logvars(enc)
        sampling = self.reparametrize(means, logvars)

        return sampling, means, logvars

    def decoder(self, x):
        dec = self.fc3(x)
        dec = F.selu(dec)
        dec = self.fc4(dec)
        dec = F.selu(dec)
        pi = self.pi(dec)
        pi = torch.sigmoid(pi)

        return pi

    def forward(self, x):
        sampling, means, logvars = self.encoder(x)
        pi = self.decoder(sampling)

        alpha = F.softplus(self.alpha) + 1.0  # alpha > 1
        theta1 = F.softplus(self.theta1)  # theta1 > 0
        theta2 = F.softplus(self.theta2)  # theta2 > 0

        return pi, alpha, theta1, theta2, means, logvars
