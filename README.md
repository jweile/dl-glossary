# The Open Source Deep Learning Glossary

Deep learning terminology can be difficult and overwhelming, especially to newcomers. This glossary tries to define the most commonly used terms. It was forked and expanded upon from https://github.com/jrdi/dl-glossary.

Since terminology is constantly changing with new terms appearing every day this glossary will be in a permanent work in progress. Feel free to make changes and submit pull requests.

To make editing and adding new entries easier, they can be found as individual markdown files in the `entries/` folder. Using the `makefile` (by calling `make`) will automatically sort, index, and collate them into this `README` file. You can also create a PDF version by calling `make buildPDF` (which requires pandoc and LaTeX to be installed).

## Table of contents

 * [Activation function](#activation-function)
 * [Affine layer](#affine-layer)
 * [Attention mechanism](#attention-mechanism)
 * [Autoencoder](#autoencoder)
 * [Average-Pooling](#average-pooling)
 * [Backpropagation](#backpropagation)
 * [Backward pass](#backward-pass)
 * [Batch normalization](#batch-normalization)
 * [Batch](#batch)
 * [Bias](#bias)
 * [Capsule Network](#capsule-network)
 * [Convolutional Neural Network (CNN)](#cnn)
 * [Context](#context)
 * [Data augmentation](#data-augmentation)
 * [Dead neuron](#dead-neuron)
 * [Decoder](#decoder)
 * [Dropout](#dropout)
 * [Embedding](#embedding)
 * [Encoder](#encoder)
 * [Epoch](#epoch)
 * [Exploding gradient](#exploding-gradient)
 * [Feed-forward](#feed-forward)
 * [Fine-tuning](#fine-tuning)
 * [Forward pass](#forward-pass)
 * [Generative Adversarial Network (GAN)](#gan)
 * [Graph Convolutional Network (GCN)](#gcn)
 * [Gradient Recurrent Unit (GRU)](#gru)
 * [Kernel](#kernel)
 * [Latent space](#latent-space)
 * [Layer](#layer)
 * [Learning rate](#learning-rate)
 * [Loss function](#loss-function)
 * [Long Short-Term Memory (LSTM)](#lstm)
 * [Max-Pooling](#max-pooling)
 * [Multi Layer Perceptron (MLP)](#mlp)
 * [Module](#module)
 * [Neuron](#neuron)
 * [Normalization layer](#normalization-layer)
 * [Pooling](#pooling)
 * [Pytorch](#pytorch)
 * [RAG](#rag)
 * [Receptive field](#receptive-field)
 * [Relational reasoning](#relational-reasoning)
 * [ReLU](#relu)
 * [Residual Networks (ResNet)](#resnet)
 * [Recurrent Neural Network (RNN)](#rnn)
 * [Sequence-to-sequence (S2S)](#s2s)
 * [Siamese Neural Network](#siamese-neural-network)
 * [Softmax](#softmax)
 * [Tensorflow](#tensorflow)
 * [Training](#training)
 * [Vanishing gradient](#vanishing-gradient)

***

## Activation function

Activation functions live inside neural network layers and modify the data they receive before passing it to the next layer. Some of them mimic the non-linear behavior of biological neurons, introducing a saturation behavior and an activation threshold. 

The choice of activation function can contribute to problems such as "[dead neurons](#dead-neuron)" as well as compute time for [forward-](forward-pass) and [backward-passes](backward-pass).

Commonly used functions include [tanh](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh), [ReLU (Rectified Linear Unit)](#relu), [sigmoid](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#sigmoid), and variants of these.

## Affine layer

A fully-connected [layer](#layer) in a Neural Network. Affine means that each neuron in the previous layer is connected to each neuron in the current layer. In many ways, this is the “standard” layer of a Neural Network. Affine layers are often added on top of the outputs of [Convolutional Neural Networks](#cnn) or [Recurrent Neural Networks](#rnn) before making a final prediction. An affine layer is typically of the form `y = f(Wx + b)` where x are the layer inputs, W the parameters, b a bias vector, and f a nonlinear [activation function](#activation-function)

## Attention mechanism

Attention Mechanisms are inspired by human attention, the ability to focus on specific parts of a larger context. Attention is the core context of the [Transformer](#transformer), in which specific features "attend to" other features. Mathematically, this is achieved by multiplying the features with "query" and "key" vectors. The 

* [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

## Autoencoder

An Autoencoder is a Neural Network model whose goal is to predict the input itself, typically through a “bottleneck” somewhere in the network. By introducing a bottleneck, we force the network to learn a lower-dimensional representation of the input, effectively compressing the input into a good representation. Autoencoders are related to PCA and other dimensionality reduction techniques, but can learn more complex mappings due to their nonlinear nature. A wide range of autoencoder architectures exist, including [Denoising Autoencoders](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), [Variational Autoencoders](http://arxiv.org/abs/1312.6114), or [Sequence Autoencoders](http://arxiv.org/abs/1511.01432).

## Average-Pooling
Average-Pooling is a [pooling](#pooling) technique used in Convolutional Neural Networks for Image Recognition. It works by sliding a window over patches of features, such as pixels, and taking the average of all values within the window. It compresses the input representation into a lower-dimensional representation.

## Backpropagation

Backpropagation is an algorithm to efficiently calculate the gradients in a Neural Network, or more generally, a feedforward computational graph. It boils down to applying the chain rule of differentiation starting from the network output and propagating the gradients backward. The first uses of backpropagation go back to Vapnik in the 1960’s, but [Learning representations by back-propagating errors](http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html) is often cited as the source.

* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)
* [Machine Learning Glossary: Backpropagation](https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html)

## Backward pass

The calculation of the [gradients](#gradient) of each model parameter relative to the [loss function]("#loss-function") for a given input data batch during [training](#training). This is done using the back-propagation algorithm. Each [batch](#batch) will have undergone a [forward pass]("forward-pass") beforehand in which the loss was calculated.

## Batch normalization

Batch Normalization is a technique that normalizes neuronal pre-activations per mini-batch. This avoids extremely large or small pre-actiations that would over- or under-saturate the activation function and thus lead to near-zero gradients. Batch normalization thus accelerates convergence by reducing internal covariate shift inside each batch. 

Batch normalization introduces additional parameters: a scale and an offset (which renders the bias term of the neuronal layer redunant).

Batch Normalization has been found to be very effective for Convolutional and Feedforward Neural Networks but hasn’t been successfully applied to Recurrent Neural Networks.

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)
* [Batch Normalized Recurrent Neural Networks](http://arxiv.org/abs/1510.01378)
* [Machine Learning Glossary: Batch Normalization](https://ml-cheatsheet.readthedocs.io/en/latest/layers.html#batchnorm)

## Batch

We can’t pass the entire dataset into the neural net at once. So, we divide dataset into Number of Batches or sets or parts.

Just like we divide a big article into multiple sets/batches/parts like Introduction, Gradient descent, Epoch, Batch size and Iterations which makes it easy to read the entire article for the reader and understand it.

## Bias

This word has multiple meanings. It could refer to the [bias term](#bias-term) or model bias.

Model bias is the average difference between the model's predictions and the correct value for that observation?

* **Low bias** could mean every prediction is correct. It could also mean half of your predictions are above their actual values and half are below, in equal proportion, resulting in low average difference.
* **High bias** (with low variance) suggests your model may be underfitting and you’re using the wrong architecture for the job.

## Capsule Network
A Capsule Neural Network (CapsNet) is a machine learning system that is a type of artificial neural network (ANN) that can be used to better model hierarchical relationships. The approach is an attempt to more closely mimic biological neural organization.

The idea is to add structures called "capsules" to a [convolutional neural network (CNN)](#cnn), and to reuse output from several of those capsules to form more stable (with respect to various perturbations) representations for higher capsules. The output is a vector consisting of the probability of an observation, and a pose for that observation. This vector is similar to what is done for example when doing classification with localization in CNNs.

* [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

## CNN

A Convolutional Neural Network (CNN) uses [convolutions](https://en.wikipedia.org/wiki/Convolution) to connected extract features from local regions of an input. Most CNNs contain a combination of convolutional, [pooling](#pooling) and [affine layers](#affine-layer). CNNs have gained popularity particularly through their excellent performance on visual recognition tasks, where they have been setting the state of the art for several years.

* [Stanford CS231n class – Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
* [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

## Context

The context of an LLM is the list of tokens that are provided to the LLM in order to generate the next token in a respons. In case of a chat-bot, this is usually the entire conversation so far up to the maximum context size. The maximum context size thus determines how much information the LLM can still take into account when generating the next response. (This is why chat-bots seem to 'lose the thread' of the conversation after a while.)

## Data augmentation

Data augmentation is the practice of artificially generating more training data from a smaller initial training set. The purpose of this practice is to provide a greater variety of training examples in a less cost- and labor-intensive manner.

For example, for image-processing models, one may add rotated, cropped, flipped or color-jittered versions of the original training images. For text data, one may replace synonyms, translate the text to another language and immediately back again ("back-translation"), paraphrasing.

## Dead neuron

A [neuron](#neuron) that never activates for any input. If this happens during [training](#training) due to its incoming [weights](#weights) always keeping it below the activation threshold, will not participate in training anymore, because its gradient has become zero. This can be avoided by choosing a different [activation function](#activation-function) or by using [normalization layers](#normalization-layer).


## Decoder

A decoder is a network [module](#module) that maps from a lower-dimensional "[latent space](latent-space)" back into its original "data-space" representation.

See also:

* [Encoder](#encoder)


## Dropout
Dropout is a regularization technique for Neural Networks that prevents overfitting. It prevents neurons from co-adapting by randomly setting a fraction of them to 0 at each training iteration. Dropout can be interpreted in various ways, such as randomly sampling from an exponential number of different networks. Dropout layers first gained popularity through their use in [CNNs](#cnn), but have since been applied to other layers, including input embeddings or recurrent networks.

* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
* [Recurrent Neural Network Regularization](http://arxiv.org/abs/1409.2329)


## Embedding

An embedding is a higher-dimensional representation of categorical inputs, such as [tokens]("#token") which their represents hidden semantic features and allows them to be compared. Inputs whose embeddings align (i.e. lie in the same direction in the multidimensional space, as measured by having a small vector dot-product) are implied to have similar semantic meaning. Data is typically translated to its embedding via a look-up table.

Contrast with:

* [Latent space]("#latent-space")

## Encoder

An encoder is a network [module](#module) that maps its input into a lower-dimensional "[latent space](latent-space)" (also called "bottleneck"), which is meant to distill the input data down to its most relevant features. 

See also:

* [Decoder](#decoder)
* [Auto-encoder](#auto-encoder)
* [Representation learning]("#representation-learning")

## Epoch
An epoch is the number of training cycles it takes to see the entire training dataset exactly one time. Conversetly, the number of training epochs is the number of times the algorithm has seen the entire data set.

## Exploding gradient
The Exploding Gradient Problem is the opposite of the [Vanishing Gradient Problem](#vanishing-gradient). In Deep Neural Networks gradients may explode during backpropagation, resulting number overflows. A common technique to deal with exploding gradients is to perform Gradient Clipping or using LeakyReLU activation function.

* [On the difficulty of training recurrent neural networks](http://arxiv.org/abs/1211.5063)

## Feed-forward
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## Fine-tuning
Fine-tuning is an example of transfer learning, in which a pre-trained model (usually a [foundation model](#foundation-model) such as BERT or GPT) is trained for additional [epochs](#epoch) on a specific task (such as answering domain-specific queries or providing specifically formatted output).

## Forward pass

The calculation of the [loss function]("#loss-function") on a given input data batch during [training](#training). The input batch is fed into the model

## GAN
## GCN
## GRU
The Gated Recurrent Unit (GRU) is a simplified version of an LSTM unit with fewer parameters. Just like an LSTM cell, it uses a gating mechanism to allow RNNs to efficiently learn long-range dependency by preventing the [vanishing gradient problem](#vanishing-gradient). The GRU consists of a reset and update gate that determine which part of the old memory to keep vs. update with new values at the current time step.

* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078v3)
* [Recurrent Neural Network Tutorial, Part 4 – Implementing a GRU/LSTM RNN with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

## Kernel
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## Latent space

A "latent space", (also called "bottleneck") is a lower-dimensional representation into which data can be mapped. It is mean to represent the key features of the underlying data, which which allows downstream components to learn more easily.

Contrast with:

* [Embedding](#embedding)

## Layer

![deep learning layers](https://miro.medium.com/max/1400/1*eJ36Jpf-DE9q5nKk67xT0Q.jpeg)

**Input Layer**

Holds the data your model will train on. Each neuron in the input layer represents a unique attribute in your dataset (e.g. height, hair color, etc.).

**Hidden Layer**

Sits between the input and output layers and applies an activation function before passing on the results. There are often multiple hidden layers in a network. In traditional networks, hidden layers are typically fully-connected layers. Each neuron receives input from all the previous layer’s neurons and sends its output to every neuron in the next layer. This contrasts with how convolutional layers work where the neurons send their output to only some of the neurons in the next layer.

**Output Layer**

The final layer in a network. It receives input from the previous hidden layer, optionally applies an activation function, and returns an output representing your model’s prediction.

## Learning rate
The size of the steps on Gradient descent is called the learning rate. With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point since the slope of the hill is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom.

## Loss function

A loss function, or cost function, is a wrapper around our model’s predict function that tells us “how good” the model is at making predictions for a given set of parameters. The loss function has its own curve and its own derivatives. The slope of this curve tells us how to change our parameters to make the model more accurate! We use the model to make predictions. We use the cost function to update our parameters. Our cost function can take a variety of forms as there are many different cost functions available. Popular loss functions include: [MSE (L2)](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mse) and [Cross-entropy Loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#loss-cross-entropy).

## LSTM
Long-short-term memory networks (LSTMs) are a form of recurrent neural network (RNN) and are motivated by the "[vanishing gr
adient problem](#vanishing-gradient)".

`TODO: Add more here.`

## Max-Pooling
A [pooling](#pooling) operations typically used in Convolutional Neural Networks. A max-pooling layer selects the maximum value from a patch of features. Just like a convolutional layer, pooling layers are parameterized by a window (patch) size and stride size. For example, we may slide a window of size 2×2 over a 10×10 feature matrix using stride size 2, selecting the max across all 4 values within each window, resulting in a new 5×5 feature matrix.

Pooling layers help to reduce the dimensionality of a representation by keeping only the most salient information, and in the case of image inputs, they provide basic invariance to translation (the same maximum values will be selected even if the image is shifted by a few pixels). Pooling layers are typically inserted between successive convolutional layers.

## MLP
## Module

A module is a functional unit in a neural network typically consisting of multiple layers which together perform pre-defined functionality, such as a [tranformer](#transformer) or an [encoder]("#encoder").

## Neuron

A neuron is the smallest working unit in a neural network or [multi-layered perceptron (MLP)](#mlp), where they are arranged in [layers](#layer). A neuron stores a single Real-valued number, called its "activation", which is calculated using a linear combination of the activations of all the neurons in the previous layer and then passed through an [activation function](#activation-function), such a s sigmoid function. The parameters of the linear combination are called the [weights](#weights) and the [bias](#bias-term).

Let $\vec{x}_{i-1}$ be the vector of actiations of the previous layer, let $W_{i}$ be the weight matrix, $\vec{b_i}$ be the vector of biases, and $\sigma$ be the activation function, then the activations of the current layer are given as:

$$ \vec{x}_i = \sigma (W_{i}\vec{x}_{i-1} + \vec{b_i}) $$

Typically the activation values of a neuron are not stored as single variables, but as a vector representing the entire layer. Similarly, the weights for the neuron are stored in a matrix, representing all the weight vectors for the layer.

### Weights

Weights are parameters of neurons. Mathematically, they are factors to be multiplied with the preceding [layer's]("#layer") [activations]("#neuron"). The weights are learned during training.

### Bias term

Bias terms are constants that act as the intercept in the linear combination calculation. Like the weights, they are learned parameters. 

## Normalization layer

A layer in a neural network that performs a normalization function (such as [batch normalization]("#batch-normalization") on [pre-activations]("#pre-activation") of the preceding linear layer.)

## Pooling
Pooling layers often take convolution layers as input. A complicated dataset with many object will require a large number of filters, each responsible finding pattern in an image so the dimensionally of convolutional layer can get large. It will cause an increase of parameters, which can lead to over-fitting. Pooling layers are methods for reducing this high dimensionally. Just like the convolution layer, there is kernel size and stride. The size of the kernel is smaller than the feature map. For most of the cases the size of the kernel will be 2X2 and the stride of 2. There are mainly two types of pooling layers, [Max-Pooling](#max-pooling) and [Average-Pooling](#average-pooling).

## Pytorch
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## RAG
Retrieval-augmented Generation (RAG) is a technique that combines a traditional querying system with an LLM. An interaction with the LLM triggers search queries against a database, repository or even the open internet to retrieve relevant data points or documents which are then "contextualized", i.e. they are worked into the LLMs [context](#context). This allows the LLM to then generate a response taking into account the new information from the search result.

## Receptive field
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## Relational reasoning
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## ReLU
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## ResNet
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## RNN
A Recurrent Neural Network (RNN) is a neural network architecture designed for processing sequential data in which the order of elements is important (i.e text or speech). The output of a neuron representing a specific timestep is fed back into the network for the next time step. 

`TODO: write more here`

## S2S

A sequence-to-sequence (S2S) architecture, is a network architecture that has sequential data (such as text) as both its input and its output. Internally, the input sequences are typically [encoded]("#encoder") in a lower-dimensional context vector. A common example would be a language translation model.

## Siamese Neural Network
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## Softmax

Softmax is a mathematical function that transforms a vector of [logits]("#logit") ( $\vec{x} \in \mathbb{R}^N$ ) into a vector of probabilities which all add up to 1.

$$ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}} $$


### Temperature 
The softmax function has a "temperature" parameter, which controls the contrast between high and low probabilities. 

## Tensorflow
Be the first to [contribute](https://github.com/jrdi/dl-glossary/pulls)!

## Training

Training is the process that allows a model to learn from labeled training data. In neural networks this is achieved through [gradient descent]("#gradient-descent") and typically proceeds in [batches]("#batch").

## Vanishing gradient
The vanishing gradient problem arises in very deep Neural Networks, typically [Recurrent Neural Networks](#rnn), that use activation functions whose gradients tend to be small (in the range of 0 from 1). Because these small gradients are multiplied during backpropagation, they tend to “vanish” throughout the layers, preventing the network from learning long-range dependencies. Common ways to counter this problem is to use activation functions like [ReLUs](#relu) that do not suffer from small gradients, or use architectures like [LSTMs](#lstm) that explicitly combat vanishing gradients. The opposite of this problem is called the [exploding gradient problem](#exploding-gradient).

* [On the difficulty of training recurrent neural networks](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)

***

## Resources

* https://ml-cheatsheet.readthedocs.io/en/latest/index.html
* https://towardsdatascience.com/the-a-z-of-ai-and-machine-learning-comprehensive-glossary-fb6f0dd8230
* http://www.wildml.com/deep-learning-glossary/
