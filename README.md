# scPDA: Denoising Protein Expression in Droplet-Based Single-Cell Data

scPDA is a VAE-based neural network for the task of denoising single-cell surface protein abundance measured by droplet-based technologies such as CITE-seq.

Unlike most currently established methods, scPDA does not require empty droplets. scPDA establishes a probabilistic model for raw count data, and shows a great computational efficiency.

For more details read our [manuscript]()
<p align="center">
  <img src="https://raw.githubusercontent.com/PancakeZoy/scPDA/refs/heads/main/img/scPDA_stru.png" width="750" title="model_pic">
</p>

## Installation
`pip install scpda`

## Demo
A [demo](demo/demo.ipynb) is available in the `demo` folder, along with the example data.

## Main API
Below is an example that includes main APIs to train `scPDA`.

```python
from scPDA import Denoiser

# please prepare the protein counts raw_counts and the estimated background mean mu1 (torch.tensor)
model = Denoiser(raw_counts=raw_counts_tensor,  bg_mean=mu1_tensor)
model.train()
model.inference()

# The background probability (model.pi) and denoised counts (model.denoised_counts) are returned
pi = model.pi
denoised_cts = model.denoised_counts
```