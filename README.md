# scPDA: Denoising Protein Expression in Droplet-Based Single-Cell Data

scPDA is a VAE-based neural network for the task of denoising single-cell surface protein abundance measured by droplet-based technologies such as CITE-seq.

Unlike most currently established methods, scPDA does not require empty droplets. scPDA establishes a probabilistic model for raw count data, and shows a great computational efficiency.

<p align="center">
  <img src="https://github.com/PancakeZoy/scPDA/blob/main/img/scPDA_stru.png?raw=true" width="750" title="model_pic">
</p>

## Installation
`pip install scpda`

## Main API
Below is an example that includes main APIs to train `scPDA` on a perturbation dataset.