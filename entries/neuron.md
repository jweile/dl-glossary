## Neuron

A neuron is the smallest working unit in a neural network or [multi-layered perceptron (MLP)](#mlp), where they are arranged in [layers](#layer). A neuron stores a single Real-valued number, called its "activation", which is calculated using a linear combination of the activations of all the neurons in the previous layer and then passed through an [activation function](#activation-function), such a s sigmoid function. The parameters of the linear combination are called the [weights](#weights) and the [bias](#bias-term).

Let $\vec{x}_{i-1}$ be the vector of actiations of the previous layer, let $W_{i}$ be the weight matrix, $\vec{b_i}$ be the vector of biases, and $\sigma$ be the activation function, then the activations of the current layer are given as:

$$ \vec{x}_i = \sigma (W_{i}\vec{x}_{i-1} + \vec{b_i}) $$

Typically the activation values of a neuron are not stored as single variables, but as a vector representing the entire layer. Similarly, the weights for the neuron are stored in a matrix, representing all the weight vectors for the layer.

### Weights

Weights are parameters of neurons. Mathematically, they are factors to be multiplied with the preceding [layer's]("#layer") [activations]("#neuron"). The weights are learned during training.

### Bias term

Bias terms are constants that act as the intercept in the linear combination calculation. Like the weights, they are learned parameters. 

