# From Autoencoders to Multimodal Generative AI

A hands-on learning repository documenting my journey through generative AI — from foundational models to multimodal LLM. Each topic includes from-scratch implementations, experiments, and detailed observations.

This is a living repository. New topics will be added as I work through them.

---

## Topics Covered

### [Autoencoder](Autoencoder/)

Standard autoencoder implementation that learns deterministic compression of input data into a fixed latent vector and reconstructs it back. Serves as the foundation for understanding latent representations before moving to probabilistic models.

- Deterministic encoder-decoder architecture
- Trained on MNIST

### [Variational Autoencoder (VAE)](VAE/)

Extends the autoencoder by mapping inputs to a *distribution* over latent space instead of a fixed vector. The encoder outputs mean and variance of an approximate posterior, and the KL divergence term regularizes it against a standard normal prior — enabling structured, sample-able latent spaces.

- From-scratch implementation with reparameterization trick
- Bernoulli and Gaussian decoder likelihoods
- Experiments across different variance parameterizations (unconstrained, clamped, global learnable)
- CNN encoder experiment on CIFAR-10 to test whether spatial feature extraction improves latent structure
- Evaluated on MNIST and CIFAR-10
- Detailed observations on training stability, reconstruction quality, and latent space structure
- [Full experiment write-up →](VAE/README.md)

### [CLIP — Contrastive Language-Image Pretraining](CLIP/)

Implementation of CLIP's contrastive learning framework that aligns image and text representations into a shared embedding space — enabling zero-shot transfer learning. Includes a from-scratch Vision Transformer and decoder-only text transformer, alongside experiments with pretrained encoders.

- From-scratch ViT + decoder-only transformer implementation
- Pretrained encoder experiments: ResNet18 + BERT, ViT-B/16 + BERT
- Temperature scaling study across fixed (0.07, 0.1, 0.5) and learnable temperatures
- InfoNCE loss (symmetric cross-entropy) with R@1 evaluation
- Trained on Flickr8k
- [Full experiment write-up →](CLIP/README.md)

---

## Roadmap

Upcoming topics as I continue this journey:

- [x] Autoencoder
- [x] VAE
- [x] CLIP
- [ ] VQ-VAE
- [ ] Multimodal Models (Vision-Language)

---

## Repository Structure

```
.
├── Autoencoder/
│   └── Autoencoder.ipynb
├── VAE/
│   ├── VAE.ipynb
│   ├── README.md
│   └── images/
├── CLIP/
│   ├── CLIP.ipynb
│   ├── README.md
│   └── images/
└── README.md
```

Each topic lives in its own directory with implementation notebooks, experiment images, and (where applicable) a dedicated README with detailed observations.
