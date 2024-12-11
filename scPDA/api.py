from typing import Optional, List, Union
from ._network import VAE
from ._loss import TotalLoss
from tqdm import tqdm
import anndata as ad
import torch
import time
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


class Denoiser:
    """
    scPDA model class for denoising protein counts data.

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
    Total_list : list
        List to store total loss per epoch during training.
    KLD_list : list
        List to store Kullback-Leibler divergence loss per epoch during training.
    Recon_list : list
        List to store reconstruction loss per epoch during training.
    runtime : float or None
        Training runtime in seconds.
    trained_model : torch.nn.Module or None
        The trained model instance.
    pi : torch.Tensor or None
        Pi parameter after training.
    alpha : torch.Tensor or None
        Alpha parameter after training.
    theta1 : torch.Tensor or None
        Theta1 parameter after training.
    theta2 : torch.Tensor or None
        Theta2 parameter after training.
    z_means : torch.Tensor or None
        Latent means after training.
    z_logvars : torch.Tensor or None
        Latent log variances after training.
    denoised_counts : torch.Tensor or None
        Denoised counts after training.
    """

    def __init__(
        self,
        raw_counts: Union[ad.AnnData, pd.DataFrame, np.ndarray, torch.Tensor],
        bg_mean: Optional[torch.Tensor] = None,
        n_components: List[int] = [2, 3],
        n_layer1: int = 100,
        n_layer2: int = 50,
        n_hidden: int = 15,
        alpha_init: Optional[torch.Tensor] = None,
        theta1_init: Optional[torch.Tensor] = None,
        theta2_init: Optional[torch.Tensor] = None,
        device: str = "auto",
    ) -> None:
        """Initialize the scPDA model.

        Parameters
        ----------
        raw_counts : torch.tensor
            A torch.tensor object contains the protein counts (cell by gene)
        bg_mean : torch.tensor, optional
            A torch.tensor object contains the estimated background mean, can be obtained from GMM
        n_components : list, optional
            List of integers specifying the number of components to try when fitting GMM.
            For each protein, the best model will be selected based on BIC.
            Default is [2, 3].
        n_layer1 : int, optional
            Number of neurons in the first hidden layer of the encoder and last hidden
            layer of decoder. Default is 100.
        n_layer2 : int, optional
            Number of neurons in the second hidden layer of encoder and first hidden
            layer of decoder. Default is 50.
        n_hidden : int, optional
            Dimensionality of the latent space (bottle neck). Default is 15.
        alpha_init : torch.Tensor or None, optional
            Initial value for alpha parameter in loss function. If None, initialized as
            torch.ones(n_proteins) * 2. If provided, must be a torch.Tensor of shape
            (n_proteins,). Default is None.
        theta1_init : torch.Tensor or None, optional
            Initial value for theta1 parameter in loss function. If None, initialized as
            torch.ones(n_proteins). If provided, must be a torch.Tensor of shape
            (n_proteins,). Default is None.
        theta2_init : torch.Tensor or None, optional
            Initial value for theta2 parameter in loss function. If None, initialized as
            torch.ones(n_proteins). If provided, must be a torch.Tensor of shape
            (n_proteins,). Default is None.
        - device:
            Device to run the model on. Default: 'auto'
        """

        # Determine the device
        if device == "auto":
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                self.device = torch.device("cuda:" + str(current_device))
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Process raw_counts input
        if isinstance(raw_counts, ad.AnnData):
            raw_counts = raw_counts.to_df()
            raw_counts = torch.tensor(raw_counts.to_numpy(), dtype=torch.float32)
        elif isinstance(raw_counts, pd.DataFrame):
            raw_counts = torch.tensor(raw_counts.to_numpy(), dtype=torch.float32)
        elif isinstance(raw_counts, np.ndarray):
            raw_counts = torch.tensor(raw_counts, dtype=torch.float32)
        elif isinstance(raw_counts, torch.Tensor):
            raw_counts = raw_counts.type(torch.float32)
        else:
            raise TypeError(
                "raw_counts must be an AnnData object, Pandas DataFrame, Numpy ndarray, or PyTorch Tensor."
            )

        # First validate bg_mean if provided with correct format
        if bg_mean is not None:
            if not isinstance(bg_mean, torch.Tensor):
                raise TypeError("bg_mean must be a torch.Tensor")
            if bg_mean.dtype != torch.float32:
                bg_mean = bg_mean.type(torch.float32)
        # If bg_mean is not provided, estimate it using GMM
        else:
            # Validate n_components
            if not isinstance(n_components, (list, tuple, np.ndarray)):
                raise TypeError("n_components must be a list, tuple, or numpy array")
            if not all(isinstance(n, int) and n > 1 for n in n_components):
                raise ValueError(
                    "All values in n_components must be integers greater than 1"
                )

            # Estimate bg_mean using GMM
            counts_np = raw_counts.numpy()
            n_proteins = counts_np.shape[1]
            bg_means = []
            for protein_idx in range(n_proteins):
                protein_data = counts_np[:, protein_idx].reshape(-1, 1)
                best_bic = float("inf")
                best_means = None
                for n_comp in n_components:
                    # fmt: off
                    gmm = GaussianMixture(n_components=n_comp, random_state=24, n_init=5).fit(protein_data)
                    # fmt: on
                    bic = gmm.bic(protein_data)
                    if bic < best_bic:
                        best_bic = bic
                        best_means = gmm.means_.flatten()
                bg_means.append(float(np.min(best_means)))
            bg_mean = torch.tensor(bg_means, dtype=torch.float32)

        # Process initialization parameters
        n_cells, n_prots = raw_counts.shape
        if alpha_init is None:
            alpha_init = torch.ones(n_prots) * 2
        elif not isinstance(alpha_init, torch.Tensor):
            raise TypeError("alpha_init must be a torch.Tensor or None")
        elif alpha_init.shape != (n_prots,):
            raise ValueError(f"alpha_init must have shape ({n_prots},)")

        if theta1_init is None:
            theta1_init = torch.ones(n_prots)
        elif not isinstance(theta1_init, torch.Tensor):
            raise TypeError("theta1_init must be a torch.Tensor or None")
        elif theta1_init.shape != (n_prots,):
            raise ValueError(f"theta1_init must have shape ({n_prots},)")

        if theta2_init is None:
            theta2_init = torch.ones(n_prots)
        elif not isinstance(theta2_init, torch.Tensor):
            raise TypeError("theta2_init must be a torch.Tensor or None")
        elif theta2_init.shape != (n_prots,):
            raise ValueError(f"theta2_init must have shape ({n_prots},)")

        # Store attributes
        # Basic data attributes
        self.raw_counts = raw_counts.to(self.device)
        self.bg_mean = bg_mean.to(self.device)
        self.n_cells = n_cells
        self.n_prots = n_prots
        # Training metrics
        self.Total_list = []
        self.KLD_list = []
        self.Recon_list = []
        self.runtime = None
        # Internal network parameters
        self._internal_params = {
            "n_layer1": n_layer1,
            "n_layer2": n_layer2,
            "n_hidden": n_hidden,
            "alpha_init": alpha_init.type(torch.float32).to(self.device),
            "theta1_init": theta1_init.type(torch.float32).to(self.device),
            "theta2_init": theta2_init.type(torch.float32).to(self.device),
        }
        # Model state attributes
        self.trained_model = None
        self.pi = None
        self.alpha = None
        self.theta1 = None
        self.theta2 = None
        self.denoised_counts = None

    def train(
        self,
        batch_size: int = 256,
        n_epochs: int = 500,
        lr: float = 0.005,
        gamma: float = 0.99,
        kld_weight: float = 0.25,
        recon_weight: float = 1.0,
        penalty_alpha: float = 0.1,
        verbose: bool = True,
    ) -> None:
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

        network = VAE(
            self.n_prots,
            layer1=self._internal_params["n_layer1"],
            layer2=self._internal_params["n_layer2"],
            n_hidden=self._internal_params["n_hidden"],
            alpha_init=self._internal_params["alpha_init"],
            theta1_init=self._internal_params["theta1_init"],
            theta2_init=self._internal_params["theta2_init"],
        ).to(self.device)

        loader = torch.utils.data.DataLoader(
            self.raw_counts, batch_size=batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        start_time = time.time()
        # Use tqdm only if verbose is True
        epochs = (
            tqdm(range(n_epochs), desc="Training", unit="epoch")
            if verbose
            else range(n_epochs)
        )
        for epoch in epochs:
            epoch_TotalLoss = 0
            epoch_kld = 0
            epoch_ReconLoss = 0

            for batch in loader:
                batch = batch.to(self.device)
                pi, alpha, theta1, theta2, means, logvars = network(batch)
                recon_loss, kld_loss, total_loss = TotalLoss(
                    batch,
                    pi,
                    self.bg_mean,
                    alpha,
                    theta1,
                    theta2,
                    means,
                    logvars,
                    kld_weight=kld_weight,
                    recon_weight=recon_weight,
                    penalty_alpha=penalty_alpha,
                )
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
        Perform inference using the trained model to obtain model parameters and denoised counts.

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
        # fmt: off
        pi, alpha, theta1, theta2, _, _ = self.trained_model(self.raw_counts)
        self.pi = pi.cpu()
        self.alpha = alpha.cpu()
        self.theta1 = theta1.cpu()
        self.theta2 = theta2.cpu()
        self.denoised_counts = ((1 - pi) * self.raw_counts).cpu()
