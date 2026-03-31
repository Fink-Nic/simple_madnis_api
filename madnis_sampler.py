# type: ignore
import numpy as np
import torch
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, Any, List, Callable, Literal
from numpy.typing import NDArray
from torch.types import Tensor

import madnis.integrator as madnis_integrator


@dataclass(kw_only=True)
class _MadnisState:
    numpy_rng_state: Dict
    torch_rng_state: Tensor
    flow_state: Dict[str, Any]
    cwnet_state: Dict[str, Any] | None
    optimizer_state: Dict[str, Any] | None
    scheduler_state: Dict[str, Any] | None
    madnis_step: int


@dataclass
class MadnisIntegrand:
    """
    disc_dims: List of the dimensions of the discrete part of the input space.
    n_cont: Dimension of the continuous part of the input space.
    eval: Integrand function.

    Function signature of the integrand:
        eval(discrete: NDArray, continuous: NDArray) -> func_val: NDArray
    shapes:
        discrete: (len(disc_dims), n_samples)
        continuous: (n_cont, n_samples)
        func_val: (n_samples,) 
    """
    disc_dims: List[int]
    n_cont: int
    eval: Callable[[NDArray, NDArray], NDArray]


@dataclass
class MadnisSampleBatch:
    """
    Sample batch return type for the MadNIS sampler.
    shapes:
        discrete: (len(disc_dims), n_samples)
        continuous: (n_cont, n_samples)
        wgt: (n_samples,)
    """
    discrete: NDArray
    continuous: NDArray
    wgt: NDArray


@dataclass
class FlowConfig:
    """
    Config for the Normalizing Flow module which generates the continuous samples.
    """
    uniform_latent: bool = True
    permutations: Literal["log"] = "log"
    layers: int = 3
    units: int = 32
    bins: int = 10
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_bin_derivative: float = 1e-3


@dataclass
class TransformerConfig:
    """
    Config for the Transformer module which generates the discrete samples.
    """
    embedding_dim: int = 64
    feedforward_dim: int = 64
    heads: int = 4
    mlp_units: int = 64
    transformer_layers: int = 1


@dataclass
class MadeConfig:
    """
    Config for the MADE module which generates the discrete samples. Alternative to Transformer (but worse).
    """
    layers: int = 2
    nodes_per_feature: int = 16


@dataclass
class MadnisConfig:
    """
    General config.
    Args:
        seed:
            Random seed for reproducibility. No Stream_ID for now.
        batch_size: 
            Number of samples per training step.
        max_batch_size: 
            Maximum number of samples to generate in one forward pass when calling get_samples(). Avoids out-of-memory errors on GPU.
        learning_rate:
            Learning rate for the optimizer.
        use_scheduler:
            If true, a learning rate scheduler will be used during training.
        scheduler_type: 
            Currently only supports "cosineannealing". Ignored if use_scheduler is False.
        loss_type: 
            Loss function to optimize during training. Options are "variance", "variance_softclip", "kl_divergence", and "kl_divergence_softclip".
        discrete_dims_position:
            Whether the sampler generates the discrete points before or after the continuous.
        discrete_model:
            Whether to use a Transformer or MADE for the discrete flow.
        flow_config:
            See ``FlowConfig`` dataclass for details
        transformer_config:
            See ``TransformerConfig`` dataclass for details
        made_config:
            See ``MadeConfig`` dataclass for details
    """
    seed: int = 42
    batch_size: int = 1024
    max_batch_size: int = 100_000
    learning_rate: float = 1e-3
    use_scheduler: bool = True
    scheduler_type: Literal["cosineannealing"] = "cosineannealing"
    loss_type: Literal["variance", "variance_softclip",
                       "kl_divergence", "kl_divergence_softclip"] = "kl_divergence"
    discrete_dims_position: Literal["first", "last"] = "first"
    discrete_model: Literal["transformer", "made"] = "transformer"
    flow_config: FlowConfig = field(default_factory=FlowConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    made_config: MadeConfig = field(default_factory=MadeConfig)


class MadnisSampler:
    def __init__(self,
                 integrand: MadnisIntegrand,
                 config: MadnisConfig,):
        import torch
        self._seed = config.seed
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(config.seed)

        self._numpy_rng = np.random.default_rng(config.seed)
        self._device = torch.device('cpu')  # default

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                if (7, 0) <= (major, minor) < (12, 0):
                    self.device = torch.device(f'cuda:{i}')
                    print(
                        f"Using CUDA device {i}: {torch.cuda.get_device_name(i)} (capability {major}.{minor})")
                    break
            else:
                print("CUDA devices found but none are compatible. Using CPU.")
        else:
            print("No CUDA device found. Using CPU.")

        self.integrand: MadnisIntegrand = integrand
        self._num_disc_dims = len(integrand.disc_dims)
        self._input_dim = self.integrand.n_cont + self._num_disc_dims
        self._dtype = np.float64

        self._use_scheduler = config.use_scheduler
        self._scheduler_type = config.scheduler_type
        self._batch_size = config.batch_size
        self._discrete_dims_position = config.discrete_dims_position

        match config.loss_type.lower():
            case "variance":
                loss = madnis_integrator.losses.stratified_variance
            case "variance_softclip":
                loss = MadnisSampler._stratified_variance_softclip
            case "kl_divergence":
                loss = madnis_integrator.losses.kl_divergence
            case "kl_divergence_softclip":
                loss = MadnisSampler._kl_divergence_softclip
            case _:
                loss = None

        madnis_integrand = madnis_integrator.Integrand(
            function=self._madnis_eval,
            input_dim=self._input_dim,
            discrete_dims=self.integrand.disc_dims,
            discrete_dims_position=self._discrete_dims_position,
            discrete_prior_prob_function=self._madnis_discrete_prior_prob_function,
        )
        match config.discrete_model.lower():
            case "transformer":
                discrete_flow_kwargs = asdict(config.transformer_config)
            case "made":
                discrete_flow_kwargs = asdict(config.made_config)
            case _:
                discrete_flow_kwargs = dict()

        self._madnis = madnis_integrator.Integrator(
            madnis_integrand,
            device=self._device,
            discrete_flow_kwargs=discrete_flow_kwargs,
            loss=loss,
            batch_size=self._batch_size,
            discrete_model=config.discrete_model,
            learning_rate=config.learning_rate,
            flow_kwargs=asdict(config.flow_config)
        )
        # self._madnis.optimizer = torch.optim.Adam(self._madnis.flow.parameters(), lr=learning_rate,
        #                                          weight_decay=1e-5, betas=(0.8, 0.99))
        self.max_batch_size = config.max_batch_size

    def train(self, n: int = 10):
        """
        Trains the sampler for n steps à config.batch_size
        Args:
            n: number of training steps
        Returns:
            None
        """
        if self._use_scheduler:
            self._madnis.scheduler = self._get_scheduler(
                n, self._scheduler_type)
        self._madnis.train(n, self._default_callback, True)

    def get_samples(self, n_samples: int) -> MadnisSampleBatch:
        """
        Args:
            n_samples: number of samples
        Returns:
            ``MadnisSampleBatch``
        """
        # LayerData objects return a view of the fields, so we can fill them directly,
        # but this would sidestep data validation, so we fill a copy and then assign it
        continuous = np.empty((self.integrand.n_cont, n_samples), dtype=self._dtype)
        discrete = np.empty((self._num_disc_dims, n_samples), dtype=np.uint64)
        wgt = np.empty((n_samples), dtype=self._dtype)

        n_eval = 0
        while n_eval < n_samples:
            n = min(self.max_batch_size, n_samples - n_eval)
            with torch.no_grad():
                x_all, prob = self._madnis.flow.sample(
                    n,
                    return_prob=True,
                    device=self._madnis.dummy.device,
                    dtype=self._madnis.dummy.dtype,
                )
            discrete[:, n_eval:n_eval+n], continuous[:, n_eval:n_eval+n] = self._madnis_output_to_disc_cont(x_all)
            wgt[n_eval:n_eval+n] = 1 / prob.numpy(force=True)
            n_eval += n

        return MadnisSampleBatch(discrete=discrete, continuous=continuous, wgt=wgt)

    def export_state(self, path: str) -> None:
        """
        Saves the Sampler state to the specified path. Will create directories, if necessary.
        Args:
            path: ``str``
        Returns:
            None
        """
        try:
            from pathlib import Path
            path: Path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            flow_state = self._madnis.flow.state_dict()
            cwnet_state = self._madnis.cwnet.state_dict() if self._madnis.cwnet is not None else None
            optimizer_state = self._madnis.optimizer.state_dict() if self._madnis.optimizer is not None else None
            scheduler_state = self._madnis.scheduler.state_dict() if self._madnis.scheduler is not None else None
            torch.save(_MadnisState(
                numpy_rng_state=self._numpy_rng.bit_generator.state,
                torch_rng_state=torch.get_rng_state(),
                flow_state=flow_state,
                cwnet_state=cwnet_state,
                optimizer_state=optimizer_state,
                scheduler_state=scheduler_state,
                madnis_step=self._madnis.step
            ), path)
        except Exception as e:
            raise ValueError(f"Error saving state to {path}: {e}")

    def import_state(self, path: str) -> None:
        """
        Imports the sampler state at ``path``. Must be called from an instance whose 
        Args:
            path: ``str``
        Returns:
            None
        Raises:
            ValueError: If the file at ``path`` is not found or is not a valid state file for MadnisSampler.
            RuntimeError: If there is an error loading the state from the file, such as an I/O error or a deserialization error.
        """
        from pathlib import Path
        path: Path = Path(path)
        if not path.is_file():
            raise ValueError(f"State file not found at {path}")
        try:
            state = torch.load(path, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Error loading state from {path}: {e}")
        if not isinstance(state, _MadnisState):
            raise ValueError("Invalid state type for MadnisSampler.")
        self._numpy_rng.bit_generator.state = state.numpy_rng_state
        torch.set_rng_state(state.torch_rng_state)
        self._madnis.flow.load_state_dict(state.flow_state)
        if state.cwnet_state is not None:
            if self._madnis.cwnet is None:
                print("WARNING: Cannot load CWNet state: Madnis integrator was not initialized with a CWNet.")
            else:
                self._madnis.cwnet.load_state_dict(state.cwnet_state)
        if state.optimizer_state is not None:
            if self._madnis.optimizer is None:
                print("WARNING: Cannot load optimizer state: Madnis integrator was not initialized with an optimizer.")
            else:
                self._madnis.optimizer.load_state_dict(state.optimizer_state)
        if state.scheduler_state is not None:
            self._madnis.scheduler = self._get_scheduler(T_max=1, scheduler_type=self._scheduler_type)
            self._madnis.scheduler.load_state_dict(state.scheduler_state)
        self._madnis.step = state.madnis_step

    def get_info(self) -> Dict[str, Any]:
        info = {
            "Input dimension": self._input_dim,
            "Continuous dimension": self.integrand.n_cont,
            "Discrete dimension": self.integrand.disc_dims,
            "Random seed": self._seed
        }
        info["Scheduler type"] = self._scheduler_type
        info["Device"] = str(self._device)
        if self._num_disc_dims > 0:
            trainable_disc_flow = sum(p.numel()
                                      for p in self._madnis.flow.discrete_flow.parameters() if p.requires_grad)
            total_disc_flow = sum(p.numel() for p in self._madnis.flow.discrete_flow.parameters())
            info["Discrete flow trainable parameters"] = trainable_disc_flow
            info["Discrete flow total parameters"] = total_disc_flow

            trainable_cont_flow = sum(p.numel()
                                      for p in self._madnis.flow.continuous_flow.parameters() if p.requires_grad)
            total_cont_flow = sum(p.numel() for p in self._madnis.flow.continuous_flow.parameters())
            info["Continuous flow trainable parameters"] = trainable_cont_flow
            info["Continuous flow total parameters"] = total_cont_flow

        trainable_flow = sum(p.numel() for p in self._madnis.flow.parameters() if p.requires_grad)
        total_flow = sum(p.numel() for p in self._madnis.flow.parameters())
        info["Flow trainable parameters"] = trainable_flow
        info["Flow total parameters"] = total_flow

        if self._madnis.cwnet is not None:
            trainable_cwnet = sum(p.numel() for p in self._madnis.cwnet.parameters() if p.requires_grad)
            total_cwnet = sum(p.numel() for p in self._madnis.cwnet.parameters())
            info["CWNet trainable parameters"] = trainable_cwnet
            info["CWNet total parameters"] = total_cwnet

        return info

    def display_info(self) -> None:
        """
        Prints the get_info() in a human-readable format to console.
        """
        info = self.get_info()
        print("MadNIS sampler information:")
        for key, value in info.items():
            print(f"    {key}: {value}")

    def free(self) -> None:
        """Performs any necessary cleanup once the sampler is no longer needed."""
        # Move modules off GPU and drop references
        if hasattr(self, "madnis") and self._madnis is not None:
            if self._device.type == 'cuda':
                try:
                    self._madnis.flow.to('cpu')
                    if self._madnis.cwnet is not None:
                        self._madnis.cwnet.to('cpu')
                except Exception:
                    # Best-effort cleanup; continue with reference release.
                    pass

            self._madnis.optimizer = None
            self._madnis.scheduler = None
            self._madnis = None

            if self._device.type == 'cuda':
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
        self.integrand = None

    def _madnis_eval(self, x_all: Tensor) -> Tensor:
        numpy_result = self.integrand.eval(
            *self._madnis_output_to_disc_cont(x_all)
        ).astype(self._dtype).flatten()

        torch_output = torch.from_numpy(
            numpy_result).to(self._device)
        return torch_output

    def _madnis_discrete_prior_prob_function(self, indices: Tensor, dim: int = 0) -> Tensor:
        """
        Implements a flat prior for the discrete model.
        """
        num_disc_input = indices.shape[1]
        if num_disc_input == self._num_disc_dims:
            return torch.zeros_like(indices, device=indices.device)

        disc_dim = self.integrand.disc_dims[num_disc_input]
        return torch.ones((len(indices), disc_dim), device=indices.device) / disc_dim

    def _madnis_output_to_disc_cont(self, x_all: Tensor) -> Tuple[NDArray, NDArray]:
        if self._discrete_dims_position == "first":
            discrete = x_all[:, :self._num_disc_dims].numpy(force=True)
            continuous = x_all[:, self._num_disc_dims:].numpy(force=True)
        else:
            discrete = x_all[:, -self._num_disc_dims:].numpy(force=True)
            continuous = x_all[:, :-self._num_disc_dims].numpy(force=True)
        return discrete.T, continuous.T

    def _get_scheduler(self, T_max: int, scheduler_type: str | None
                       ) -> torch.optim.lr_scheduler._LRScheduler | None:
        if scheduler_type is None:
            return None
        match scheduler_type.lower():
            case 'cosineannealing':
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self._madnis.optimizer, T_max=T_max)
            case 'reducelronplateau':
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._madnis.optimizer, T_max=T_max)
            case 'linear':
                return torch.optim.lr_scheduler.LinearLR(
                    self._madnis.optimizer, T_max=T_max)
            case _:
                return None

    @staticmethod
    def _default_callback(status: madnis_integrator.TrainingStatus) -> None:
        if (status.step + 1) % 10 == 0:
            print(f"Step {status.step+1}: Loss={status.loss} ")

    @staticmethod
    def _softclip(x: torch.Tensor, threshold: torch.Tensor = 30.0):
        return threshold * torch.arcsinh(x / threshold)

    @staticmethod
    def _stratified_variance_softclip(
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor | None = None,
        channels: torch.Tensor | None = None,
        threshold: torch.Tensor = 30.0,
    ):
        """
        Computes the stratified variance as introduced in [2311.01548] for two given sets of
        probabilities, ``f_true`` and ``q_test``. It uses importance sampling with a sampling
        probability specified by ``q_sample``. A soft clipping function is applied to the
        sample weights.

        Args:
            f_true: normalized integrand values
            q_test: estimated function/probability
            q_sample: sampling probability
            channels: channel indices or None in the single-channel case
            threshold: approximate point of transition between linear and logarithmic behavior
        Returns:
            computed stratified variance
        """
        if q_sample is None:
            q_sample = q_test
        if channels is None:
            norm = torch.mean(f_true.detach().abs() / q_sample)
            f_true = MadnisSampler.softclip(
                f_true / q_sample / norm, threshold) * q_sample * norm
            abs_integral = torch.mean(f_true.detach().abs() / q_sample)
            return madnis_integrator.losses._variance(f_true, q_test, q_sample) / abs_integral.square()

        stddev_sum = 0
        abs_integral = 0
        for i in channels.unique():
            mask = channels == i
            fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
            norm = torch.mean(fi.detach().abs() / qsi)
            fi = MadnisSampler._softclip(
                fi / qsi / norm, threshold) * qsi * norm
            stddev_sum += torch.sqrt(madnis_integrator.losses._variance(fi, qti,
                                     qsi) + madnis_integrator.losses.dtype_epsilon(f_true))
            abs_integral += torch.mean(fi.detach().abs() / qsi)
        return (stddev_sum / abs_integral) ** 2

    @staticmethod
    @madnis_integrator.losses.multi_channel_loss
    def _kl_divergence_softclip(
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        threshold: torch.Tensor = 30.0,
    ) -> torch.Tensor:
        """
        Computes the Kullback-Leibler divergence for two given sets of probabilities, ``f_true`` and
        ``q_test``. It uses importance sampling, i.e. the estimator is divided by an additional factor
        of ``q_sample``. A soft clipping function is applied to the sample weights.

        Args:
            f_true: normalized integrand values
            q_test: estimated function/probability
            q_sample: sampling probability
            channels: channel indices or None in the single-channel case
            threshold: approximate point of transition between linear and logarithmic behavior
        Returns:
            computed KL divergence
        """
        f_true = f_true.detach().abs()
        weight = f_true / q_sample
        weight /= weight.abs().mean()
        clipped_weight = MadnisSampler._softclip(weight, threshold)
        log_q = torch.log(q_test)
        log_f = torch.log(clipped_weight * q_sample +
                          madnis_integrator.losses.dtype_epsilon(f_true))
        return torch.mean(clipped_weight * (log_f - log_q))
