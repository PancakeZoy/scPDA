import torch
from torch.distributions import Normal, kl_divergence


def loss_likelihood(x, pi, mu1, alpha, theta1, theta2):
    eta = 1e-10

    mu1 = mu1.expand_as(pi)
    alpha = alpha.expand_as(pi)
    mu2 = mu1 * alpha
    theta1 = theta1.expand_as(pi)
    theta2 = theta2.expand_as(pi)

    comp1_t1 = (
        torch.lgamma(x + theta1 + eta)
        - torch.lgamma(x + 1 + eta)
        - torch.lgamma(theta1 + eta)
    )
    comp1_t2 = theta1 * torch.log(theta1 / (theta1 + mu1) + eta) + x * torch.log(
        mu1 / (theta1 + mu1) + eta
    )
    NB1 = torch.exp(comp1_t1 + comp1_t2)

    comp2_t1 = (
        torch.lgamma(x + theta2 + eta)
        - torch.lgamma(x + 1 + eta)
        - torch.lgamma(theta2 + eta)
    )
    comp2_t2 = theta2 * torch.log(theta2 / (theta2 + mu2) + eta) + x * torch.log(
        mu2 / (theta2 + mu2) + eta
    )
    NB2 = torch.exp(comp2_t1 + comp2_t2)

    MNB = pi * NB1 + (1 - pi) * NB2
    loss = -torch.log(MNB + 1e-10)

    return torch.mean(loss)


def kld(means, logvars):
    std = torch.exp(0.5 * logvars)
    mean_n01 = torch.zeros_like(means)
    std_n01 = torch.ones_like(std)
    kld_loss = kl_divergence(Normal(means, std), Normal(mean_n01, std_n01))

    return torch.mean(kld_loss)


def TotalLoss(
    x,
    pi,
    mu1,
    alpha,
    theta1,
    theta2,
    means,
    logvars,
    kld_weight=0.1,
    recon_weight=1,
    penalty_alpha=0.03,
):
    recon_loss = loss_likelihood(x, pi, mu1, alpha, theta1, theta2)
    kld_loss = kld(means, logvars)
    penalty = -penalty_alpha * torch.mean(torch.log(alpha))
    total_loss = recon_loss * recon_weight + kld_loss * kld_weight + penalty

    return recon_loss, kld_loss, total_loss
