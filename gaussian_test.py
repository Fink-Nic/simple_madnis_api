from madnis_sampler import (
    MadnisIntegrand,
    MadnisConfig,
    FlowConfig,
    TransformerConfig,
    MadeConfig,
    MadnisSampler
)
import numpy as np
from numpy.typing import NDArray
from pathlib import Path


def gaussian_eval(discrete: NDArray, continuous: NDArray) -> NDArray:
    # Spherical param
    x, y, z = np.vsplit(continuous, [1, 2])

    r = x/(1-x)
    cos_az = (2*y-1)
    sin_az = np.sqrt(1 - cos_az**2)
    pol = 2*np.pi*z

    momentum = r * np.vstack(
        [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
    jac = 4*np.pi * x**2 / (1 - x)**4

    # Gaussian with sigma=1.0, normalized to integrate to 1 over the whole space
    sigma = 1.0
    norm_factor = (2*np.pi * sigma ** 2)**(momentum.shape[0]/2)

    return jac*np.exp(-(momentum**2).sum(axis=0) / sigma**2 / 2) / norm_factor


if __name__ == "__main__":
    # Testing exposed functionality
    integrand = MadnisIntegrand(
        disc_dims=[2, 3],
        n_cont=3,
        eval=gaussian_eval
    )
    config = MadnisConfig(
        seed=42,
        batch_size=1024,
        max_batch_size=10_000,
        learning_rate=1e-3,
        use_scheduler=True,
        scheduler_type="cosineannealing",
        loss_type="kl_divergence",
        discrete_dims_position="first",
        discrete_model="transformer",
        flow_config=FlowConfig(
            uniform_latent=True,
            permutations="log",
            layers=3,
            units=32,
            bins=10,
            min_bin_width=1e-3,
            min_bin_height=1e-3,
            min_bin_derivative=1e-3
        ),
        transformer_config=TransformerConfig(
            embedding_dim=64,
            feedforward_dim=64,
            heads=4,
            mlp_units=64,
            transformer_layers=1
        ),
        made_config=MadeConfig(
            layers=2,
            nodes_per_feature=16
        )
    )
    sampler = MadnisSampler(integrand, config)
    sampler.display_info()
    print("Getting initial samples...")
    samples = sampler.get_samples(10_000)
    print("Discrete samples shape:", samples.discrete.shape)
    print("Continuous samples shape:", samples.continuous.shape)
    print("Weights shape:", samples.wgt.shape)
    res = gaussian_eval(samples.discrete, samples.continuous)*samples.wgt
    res /= np.prod(np.array(integrand.disc_dims))
    print(f"Result before training: {res.mean()} +- {res.std() / np.sqrt(1000)}  TARGET: 1.0")
    print("Testing export of state...")
    state_path = Path("madnis_state.pt")
    sampler.export_state(state_path)
    print("Starting training...")
    sampler.train(n=10)
    print("Testing import of state...")
    sampler.import_state(state_path)
    if state_path.exists():
        state_path.unlink()  # Clean up the saved state file
        print("Saved state file removed.")
    else:
        print("Warning: Saved state file not found for cleanup.")

    sampler.train(n=10)
    print("Training completed. Getting samples...")
    samples = sampler.get_samples(10_000)
    res = gaussian_eval(samples.discrete, samples.continuous)*samples.wgt
    res /= np.prod(np.array(integrand.disc_dims))
    print(f"Result after training: {res.mean()} +- {res.std() / np.sqrt(1000)}  TARGET: 1.0")
    sampler.free()
    print("Sampler resources cleaned up.")
