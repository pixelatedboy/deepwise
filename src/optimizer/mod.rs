pub mod sgd;
use sgd::SGD;

pub trait Optimizer {
    fn update(&mut self, neuron: &mut Neuron, grad_weights: &[f64], grad_bias: f64, neuron_id: String);
}
