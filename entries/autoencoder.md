## Autoencoder

An Autoencoder is a Neural Network model whose goal is to predict the input itself, typically through a “bottleneck” somewhere in the network. By introducing a bottleneck, we force the network to learn a lower-dimensional representation of the input, effectively compressing the input into a good representation. Autoencoders are related to PCA and other dimensionality reduction techniques, but can learn more complex mappings due to their nonlinear nature. A wide range of autoencoder architectures exist, including [Denoising Autoencoders](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), [Variational Autoencoders](http://arxiv.org/abs/1312.6114), or [Sequence Autoencoders](http://arxiv.org/abs/1511.01432).

