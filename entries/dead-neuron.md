## Dead neuron

A [neuron](#neuron) that never activates for any input. If this happens during [training](#training) due to its incoming [weights](#weights) always keeping it below the activation threshold, will not participate in training anymore, because its gradient has become zero. This can be avoided by choosing a different [activation function](#activation-function) or by using [normalization layers](#normalization-layer).


