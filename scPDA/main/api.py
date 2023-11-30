from ._network import VAE
from ._loss import TotalLoss
from tqdm import tqdm
import anndata as ad
import torch
import time
import pandas as pd
import numpy as np


class model():
	def __init__(
		self, 
		raw_counts,
		bg_mean,
		n_layer1=100, 
		n_layer2=50, 
		n_hidden=15,
		alpha_init=None, 
		theta1_init=None, 
		theta2_init=None
	):
		if isinstance(raw_counts, ad.AnnData):
			# convert AnnData to tensor
			raw_counts = raw_counts.to_df()
			raw_counts = torch.tensor(raw_counts.to_numpy(), dtype=torch.float32)
		elif isinstance(raw_counts, pd.DataFrame):
			# convert pandas DataFrame to tensor
			raw_counts = torch.tensor(raw_counts.to_numpy(), dtype=torch.float32)
		elif isinstance(raw_counts, np.ndarray):
			# convert numpy ndarray to tensor
			raw_counts = torch.tensor(raw_counts, dtype=torch.float32)
		elif isinstance(raw_counts, torch.Tensor):
			# Ensure the tensor is of the correct type
			if raw_counts.dtype != torch.float32:
				raw_counts = raw_counts.type(torch.float32)
		else:
			raise TypeError("raw_counts must be an AnnData object, Pandas DataFrame, Numpy ndarray, or PyTorch Tensor.")

		self.raw_counts = raw_counts
		self.bg_mean = bg_mean
		self.n_layer1 = n_layer1
		self.n_layer2 = n_layer2
		self.n_hidden = n_hidden
		self.alpha_init = alpha_init
		self.theta1_init = theta1_init
		self.theta2_init = theta2_init

		self.n_cells, self.n_prots = raw_counts.shape
		self.Total_list = []
		self.KLD_list = []
		self.Recon_list = []
		self.runtime = None

		self.trained_model = None
		self.pi = None
		self.mu1 = bg_mean
		self.alpha = None
		self.theta1 = None
		self.theta2 = None
		self.z_means = None
		self.z_logvars = None
		self.denoised_counts = None



	def train(
		self,
		batch_size=256, 
		n_epochs=500, 
		lr=0.005, 
		gamma=0.99, 
		kld_weight=0.25, 
		recon_weight=1., 
		penalty_alpha=0.1, 
		verbose=True
		):

		network = VAE(self.n_prots, 
					layer1=self.n_layer1, 
					layer2=self.n_layer2, 
					n_hidden=self.n_hidden, 
					alpha_init=self.alpha_init, 
					theta1_init=self.theta1_init, 
					theta2_init=self.theta2_init
					)

		loader = torch.utils.data.DataLoader(self.raw_counts, batch_size=batch_size, shuffle=True)
		optimizer = torch.optim.Adam(network.parameters(), lr=lr)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

		start_time = time.time()
		# Use tqdm only if verbose is True
		epochs = tqdm(range(n_epochs), desc='Training', unit='epoch') if verbose else range(n_epochs)
		for epoch in epochs:
			epoch_TotalLoss = 0
			epoch_kld = 0
			epoch_ReconLoss = 0

			for batch in loader:
				pi, alpha, theta1, theta2, means, logvars = network(batch)
				recon_loss, kld_loss, total_loss = TotalLoss(batch, pi, self.bg_mean, alpha, theta1, theta2, means, logvars, 
															kld_weight=kld_weight, recon_weight=recon_weight, penalty_alpha=penalty_alpha)
				total_loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				epoch_TotalLoss += total_loss.item()
				epoch_kld += kld_loss.item()
				epoch_ReconLoss += recon_loss.item()

			# Average the epoch loss
			epoch_TotalLoss /= len(loader)
			epoch_kld /= len(loader)
			epoch_ReconLoss /= len(loader)

			# Append the loss to the loss list
			self.Total_list.append(epoch_TotalLoss)
			self.KLD_list.append(epoch_kld)
			self.Recon_list.append(epoch_ReconLoss)

			# Step the learning rate scheduler
			scheduler.step()

		self.trained_model = network
		self.runtime = time.time() - start_time

	@torch.no_grad()
	def inference(self):
		self.pi, self.alpha, self.theta1, self.theta2, self.z_means, self.z_logvars = self.trained_model(self.raw_counts)
		self.denoised_counts = (1-self.pi) * self.raw_counts