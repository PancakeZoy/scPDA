# scPDA: Denoising Protein Expression in Droplet-Based Single-Cell Data

scPDA is a VAE-based neural network for the task of denoising single-cell surface protein abundance measured by droplet-based technologies such as CITE-seq.

Unlike most currently established methods, scPDA does not require empty droplets. scPDA establishes a probabilistic model for raw count data, and shows a great computational efficiency.

<p align="center">
  <img src="https://raw.githubusercontent.com/PancakeZoy/scPDA/refs/heads/main/img/scPDA_stru.png?token=GHSAT0AAAAAACX3BPPB2BMQPALUYU4M2NJSZXR7HLA" width="750" title="model_pic">
</p>

## Installation
`pip install scpda`

## Main API
Below is an example that includes main APIs to train `scPDA`.

```python
from scPDA import model

# please prepare the protein counts dsb_counts_tensor (torch.tensor) and the estimated background mean dsb_mu1_tensor (torch.tensor)
scPDA = model(raw_counts=dsb_counts_tensor, bg_mean=dsb_mu1_tensor)
scPDA.train()
scPDA.inference()

# The estimated mu1, mu2, theta1, theta2, pi (background probability) are returned
mu1 = scPDA.mu1
mu2 = scPDA.mu1 * scPDA.alpha
theta1 = scPDA.theta1
theta2 = scPDA.theta2
pi = scPDA.pi
```