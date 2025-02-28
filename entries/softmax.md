## Softmax

Softmax is a mathematical function that transforms a vector of [logits]("#logit") ( $\vec{x} \in \mathbb{R}^N$ ) into a vector of probabilities which all add up to 1.

$$ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}} $$


### Temperature 
The softmax function has a "temperature" parameter, which controls the contrast between high and low probabilities. 

