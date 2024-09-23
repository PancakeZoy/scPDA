from ._network import VAE
from ._loss import TotalLoss
from tqdm import tqdm
import anndata as ad
import torch
import time
import pandas as pd
import numpy as np


class model():
	"""
	scPDA model class

	Parameters
	----------
	raw_counts: torch.tensor
		A torch.tensor object contains the protein counts (cell by gene)
	bg_mean: torch.tensor
		A torch.tensor object contains the estimated background mean, can be easily obtained from Gaussian Mixture Model (GMM)
	n_layer1 : int, optional
		Number of neurons in the first hidden layer of the encoder and the last hidden layer of the deocder. Default is 100.
	n_layer2 : int, optional
		Number of neurons in the second hidden layer of the encoder and the first hidden lyaer of the decoder. Default is 50.
	n_hidden : int, optional
		Dimensionality of the latent space (bottle neck). Default is 15.
	alpha_init : torch.Tensor or None, optional
		Initial value for the alpha parameter in the loss function. If None, it will be initialized as 1 during training.
	theta1_init : torch.Tensor or None, optional
		Initial value for the theta1 parameter in the loss function. If None, it will be initialized as 1 during training.
	theta2_init : torch.Tensor or None, optional
		Initial value for the theta2 parameter in the loss function. If None, it will be initialized as 1 during training.        

	Attributes
	----------
	raw_counts : torch.Tensor
		The processed raw counts data in tensor form.
	bg_mean : torch.Tensor or numpy.ndarray
		Background mean protein counts.
	n_cells : int
		Number of cells in the dataset.
	n_prots : int
		Number of proteins (features) in the dataset.
	n_layer1 : int
		Number of neurons in the first hidden layer.
	n_layer2 : int
		Number of neurons in the second hidden layer.
	n_hidden : int
		Dimensionality of the latent space.
	alpha_init : torch.Tensor or None
		Initial alpha parameter.
	theta1_init : torch.Tensor or None
		Initial theta1 parameter.
	theta2_init : torch.Tensor or None
		Initial theta2 parameter.
	Total_list : list
		List to store total loss per epoch during training.
	KLD_list : list
		List to store Kullback-Leibler divergence loss per epoch during training.
	Recon_list : list
		List to store reconstruction loss per epoch during training.
	runtime : float or None
		Training runtime in seconds. Initialized as None.
	trained_model : torch.nn.Module or None
		Placeholder for the trained model. Initialized as None.
	pi : torch.Tensor or None
		Placeholder for the pi parameter after training.
	mu1 : torch.Tensor or numpy.ndarray
		Background mean protein counts (same as `bg_mean`).
	alpha : torch.Tensor or None
		Placeholder for the alpha parameter after training.
	theta1 : torch.Tensor or None
		Placeholder for the theta1 parameter after training.
	theta2 : torch.Tensor or None
		Placeholder for the theta2 parameter after training.
	z_means : torch.Tensor or None
		Placeholder for latent means after training.
	z_logvars : torch.Tensor or None
		Placeholder for latent log variances after training.
	denoised_counts : torch.Tensor or None
		Placeholder for denoised counts after training.
	"""
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
        
		"""
		Train the scPDA model to denoise the raw protein counts data.
	
		Parameters
		----------
		batch_size : int, optional
			The number of samples per batch during training. Default is 256.
		n_epochs : int, optional
			The number of epochs to train the model. Default is 500.
		lr : float, optional
			Learning rate for the optimizer. Default is 0.005.
		gamma : float, optional
			Multiplicative factor for learning rate decay in the scheduler. Default is 0.99.
		kld_weight : float, optional
			Weight for the Kullback-Leibler divergence (KLD) loss component. Default is 0.25.
		recon_weight : float, optional
			Weight for the reconstruction loss component. Default is 1.0.
		penalty_alpha : float, optional
			Regularization weight for the alpha parameter in the loss function. Default is 0.1.
		verbose : bool, optional
			If True, displays a progress bar during training. Default is True.
	
		Returns
		-------
		None
	
		Updates
		-------
		self.trained_model : torch.nn.Module
			The trained scPDA model instance.
		self.runtime : float
			Total time taken for training, in seconds.
		self.Total_list : list
			List of average total losses for each epoch.
		self.KLD_list : list
			List of average KLD losses for each epoch.
		self.Recon_list : list
			List of average reconstruction losses for each epoch.
	
		Notes
		-----
		- Initializes the VAE model with specified architecture parameters.
		- Uses Adam optimizer and ExponentialLR scheduler for training.
		- Calculates total loss combining reconstruction loss, KLD loss, and a penalty term.
		- Records training loss metrics and runtime for analysis.
		"""
        
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
		"""        
		Updates
		-------
    	pi : torch.Tensor or None
    		Placeholder for the pi parameter after training.
    	alpha : torch.Tensor or None
    		Placeholder for the alpha parameter after training.
    	theta1 : torch.Tensor or None
    		Placeholder for the theta1 parameter after training.
    	theta2 : torch.Tensor or None
    		Placeholder for the theta2 parameter after training.
    	z_means : torch.Tensor or None
    		Placeholder for latent means after training.
    	z_logvars : torch.Tensor or None
    		Placeholder for latent log variances after training.
    	denoised_counts : torch.Tensor or None
    		Placeholder for denoised counts after training.
		"""
		self.pi, self.alpha, self.theta1, self.theta2, self.z_means, self.z_logvars = self.trained_model(self.raw_counts)
		self.denoised_counts = (1-self.pi) * self.raw_counts