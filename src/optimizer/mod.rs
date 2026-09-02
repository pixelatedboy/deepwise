use crate::nn::neuron::Neuron;

pub trait Optimizer {
    fn update(&mut self, neuron: &mut Neuron, grad_weights: &[f64], grad_bias: f64, neuron_id: &str);
}

pub mod sgd;
pub use sgd::SGD;

pub mod adam;
pub use adam::Adam;
