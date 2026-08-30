use crate::nn::neuron::Neuron;
use crate::optimizer::Optimizer;

pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        SGD { learning_rate: lr }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, neuron: &mut Neuron, grad_weights: &[f64], grad_bias: f64, _neuron_id: &str) {
        if neuron.frozen { return; }
        for i in 0..neuron.weights.len() {
            neuron.weights[i] -= self.learning_rate * grad_weights[i];
        }
        neuron.bias -= self.learning_rate * grad_bias;
    }
}
