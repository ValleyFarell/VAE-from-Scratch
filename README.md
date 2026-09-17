# Variational Autoencoders

Educational PyTorch VAE implementations for Two Moons and MNIST, with the
reparameterization trick and the standard negative ELBO objective.

- **Two Moons:** a 2D latent space and Gaussian observation model.
- **MNIST:** a 16D latent space and Bernoulli observation model with dynamic binarization.
- Training curves, held-out reconstructions, prior samples, and latent interpolation.

## Usage

Open `VariationalAutoencoder_MNIST.ipynb` in Google Colab or Jupyter and run the cells
in order. A GPU is recommended. Both models train from scratch; MNIST downloads automatically.

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

The notebook distinguishes decoder probabilities from actual Bernoulli samples.

## Sources

- Kingma and Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
