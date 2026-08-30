use crate::nn::neuron::Neuron;
use crate::functional::activation::Activation;
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
use rand::rng;

#[derive(Debug, Clone)]
pub struct Linear {
    pub neurons: Vec<Neuron>,
    dropout_rate: f64,
    mask: Option<Vec<usize>>,
    training_mode: bool,
    sampling_rate: f64,
    sampling_scale: bool,
    sampling_strategy: String,
    sampling_mask: Option<Vec<usize>>,
}

impl Linear {
    pub fn new(num_inputs: usize, num_neurons: usize, activation: Activation,
               dropout_rate: f64, sampling_rate: f64) -> Self {
                   let neurons = (0..num_neurons)
                   .map(|_| Neuron::new(num_inputs, activation))
                   .collect();
                   Linear {
                       neurons,
                       dropout_rate,
                       mask: None,
                       training_mode: false,
                       sampling_rate,
                       sampling_scale: true,
                       sampling_strategy: "random".to_string(),
                       sampling_mask: None,
                   }
               }

               pub fn forward(&mut self, inputs: &[f64], training: bool) -> Vec<f64> {
                   self.training_mode = training;
                   let mut outputs: Vec<f64> = self.neurons
                   .iter_mut()
                   .map(|n| n.forward(inputs))
                   .collect();

                   // Dropout
                   if training && self.dropout_rate > 0.0 {
                       let mut rng = rand::rng();
                       let active: Vec<usize> = (0..outputs.len())
                       .filter(|_| rng.random::<f64>() >= self.dropout_rate) // raw identifier
                       .collect();
                       self.mask = Some(active.clone());
                       let scale = 1.0 / (1.0 - self.dropout_rate);
                       let mut new_out = vec![0.0; outputs.len()];
                       for &idx in &active {
                           new_out[idx] = outputs[idx] * scale;
                       }
                       outputs = new_out;
                   } else {
                       self.mask = None;
                   }

                   // Sampling
                   if training && self.sampling_rate > 0.0 && self.sampling_rate < 1.0 {
                       let keep = ((self.neurons.len() as f64 * (1.0 - self.sampling_rate)).round() as usize).max(1);
                       let mut indices: Vec<usize> = (0..self.neurons.len()).collect();
                       let mut rng = rng();
                       indices.shuffle(&mut rng);
                       indices.truncate(keep);
                       self.sampling_mask = Some(indices.clone());
                       let scale = if self.sampling_scale && self.sampling_rate < 1.0 {
                           1.0 / (1.0 - self.sampling_rate)
                       } else { 1.0 };
                       let mut new_out = vec![0.0; outputs.len()];
                       for &idx in &indices {
                           new_out[idx] = outputs[idx] * scale;
                       }
                       outputs = new_out;
                   } else {
                       self.sampling_mask = None;
                   }

                   outputs
               }

               pub fn backward(&mut self, d_outputs: &mut [f64], is_last_layer: bool) -> (Vec<f64>, Vec<(Vec<f64>, f64)>) {
                   if self.training_mode {
                       // undo sampling
                       if let Some(mask) = &self.sampling_mask {
                           let scale = if self.sampling_scale && self.sampling_rate < 1.0 {
                               1.0 / (1.0 - self.sampling_rate)
                           } else { 1.0 };
                           for i in 0..d_outputs.len() {
                               if !mask.contains(&i) {
                                   d_outputs[i] = 0.0;
                               } else {
                                   d_outputs[i] *= scale;
                               }
                           }
                       }
                       // undo dropout
                       if let Some(mask) = &self.mask {
                           let scale = 1.0 / (1.0 - self.dropout_rate);
                           for i in 0..d_outputs.len() {
                               d_outputs[i] *= if mask.contains(&i) { scale } else { 0.0 };
                           }
                       }
                   }

                   let mut neuron_grads = Vec::with_capacity(self.neurons.len());
                   let mut d_inputs = vec![0.0; self.neurons[0].weights.len()];

                   for (i, neuron) in self.neurons.iter_mut().enumerate() {
                       let (grad_w, grad_b, d_in) = neuron.backward(d_outputs[i], is_last_layer);
                       neuron_grads.push((grad_w, grad_b));
                       for (j, val) in d_in.iter().enumerate() {
                           d_inputs[j] += val;
                       }
                   }

                   (d_inputs, neuron_grads)
               }

               pub fn freeze_neuron(&mut self, idx: usize) {
                   if idx < self.neurons.len() { self.neurons[idx].frozen = true; }
               }
               pub fn unfreeze_neuron(&mut self, idx: usize) {
                   if idx < self.neurons.len() { self.neurons[idx].frozen = false; }
               }
               pub fn freeze_all(&mut self) {
                   for n in &mut self.neurons { n.frozen = true; }
               }
               pub fn unfreeze_all(&mut self) {
                   for n in &mut self.neurons { n.frozen = false; }
               }
               pub fn get_frozen_mask(&self) -> Vec<bool> {
                   self.neurons.iter().map(|n| n.frozen).collect()
               }
}
