# From Autoencoders to Multimodal Generative AI

A hands-on learning repository documenting my journey through generative AI, from foundational models to autoregressive image generation. Each topic includes from-scratch implementations, experiments and detailed observations.

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

### [VQ-VAE — Vector Quantized Variational Autoencoder](VQVAE/)

From-scratch implementation of VQ-VAE to understand discrete latent representations after reading the "Neural Discrete Representation Learning" paper. Explores why discrete latent spaces overcome VAE's posterior collapse and weak prior problems, and why learning a prior over discrete codes is easier than over continuous space.

- Three codebook update strategies: gradient descent, EMA, and EMA + codebook restart
- 8 experiments investigating codebook collapse, utilization, and reconstruction quality
- Background on VAE failure modes (posterior collapse, weak prior, aggregate posterior)
- Evaluated on CIFAR-10 with LPIPS and rFID
- [Full experiment write-up →](VQVAE/README.md)

### [VQ-VAE-2 — Hierarchical Vector Quantized Variational Autoencoder](VQVAE2/)

Implementation of VQ-VAE-2 to explore hierarchical discrete latent representations. Investigates how splitting the latent space into top (global) and bottom (local) levels improves reconstruction quality and separates structural information from fine-grained details.

- Hierarchical architecture with top and bottom codebooks
- 7 experiments across two sets (with and without perceptual loss)
- Investigation into the role of latent dimension in hierarchical collapse
- Perceptual loss (LPIPS) integration and its impact on sharpness
- Evaluated on Imagenette with LPIPS and rFID
- [Full experiment write-up →](VQVAE2/README.md)

### [Autoregressive Image Gen (DALL·E 1 Style)](Autoregressive_Image_Gen/)

From-scratch implementation of autoregressive text-to-image generation based on the DALL·E 1 paper. Instead of modeling complex architectures or auxiliary losses directly over image pixels, this project leverages a simple approach: quantizing images into discrete codes using a VQ-VAE, concatenating text and image tokens, and training a decoder-only transformer to autoregressively model the joint distribution p(x, y).

- Stage 1: VQ-VAE image tokenizer (downsampling images to 32×32 discrete code grids)
- Stage 2: Transformer prior (conditioned on GPT-2 tokenized captions)
- 4 sequential experiments focusing on local vs. full causal attention masks, and training scale
- Validation via intentional overfitting on a small 100-image subset to prove pipeline correctness
- [Full experiment write-up →](Autoregressive_Image_Gen/README.md)

---

## Roadmap

Upcoming topics as I continue this journey:

- [x] Autoencoder
- [x] VAE
- [x] CLIP
- [x] VQ-VAE
- [x] VQ-VAE-2
- [x] Autoregressive Image Gen (DALL·E 1)

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
├── VQVAE/
│   ├── VQVAE.ipynb
│   ├── README.md
│   └── images/
├── VQVAE2/
│   ├── VQ_VAE_2.ipynb
│   ├── README.md
│   └── images/
├── Autoregressive_Image_Gen/
│   ├── Autoregressive_image_generation_latest.ipynb
│   ├── README.md
│   └── result.png
└── README.md
```

Each topic lives in its own directory with implementation notebooks, experiment images, and (where applicable) a dedicated README with detailed observations.

