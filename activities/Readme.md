## Learning objectives

- `Lecture1_NN_torch.ipynb`: for newcomers and those wanting practical experience training models
    - Experience simple data processing for an ML task
    - Engage with building and training simple neural networks (NNs) using `torch`
    -  Begin building intuition for tuning NNs
- `Lecture1-NN-JAX.ipynb`: for those interested in the fundamental concepts and math
    - Understand the mathematics of NN architectures through explicit forward-propagation code
    - Gain experience with JAX for gradient-descent optimization in NN training

- `Lecture1-NN-keras.ipynb`: TensorFlow/Keras version of the first neural-network activity
    - Download the particle dataset directly in Colab
    - Build and tune a dense regression network

- `Lecture2-CNN-torch.ipynb`: current convolutional neural-network activity using PyTorch
    - Build and train a small CNN on MNIST
    - Apply transfer learning to AT-TPC images with VGG16
    - Use `/255` scaling that preserves zero-valued pixels

- `Lecture2-CNN.ipynb`: TensorFlow/Keras version of the Lecture 2 activity
    - Based on the December 2024 versioning update from `hsf-india-deep-learning`
    - Loads only the simulated subset used in the activity
    - Repeats grayscale channels per batch rather than copying the full dataset

- `Lecture3-VAE.ipynb`: current TensorFlow/Keras variational-autoencoder activity
    - Train a genuine VAE on MNIST with reconstruction and KL-divergence losses
    - Visualize and generate samples from a two-dimensional latent space
    - Extend the architecture to AT-TPC images using a Colab-safe 10,000-event subset
