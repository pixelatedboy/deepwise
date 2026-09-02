use crate::nn::linear::Linear;
use crate::optimizer::{Optimizer, SGD, Adam};
use crate::functional::activation::Activation;
use rand::seq::SliceRandom;
use rand::rng;

#[derive(Debug, Clone, PartialEq)]
pub enum Task {
    Binary,
    Multi,
    Regression,
}

#[derive(Debug, Clone)]
pub struct Network {
    pub layers: Vec<Linear>,
    pub task: Task,
}

impl Network {
    pub fn new(layer_sizes: &[usize], activations: &[Activation], task: Task,
               dropout_rates: &[f64], sampling_rates: &[f64]) -> Self {
                   let n = layer_sizes.len() - 1;
                   let mut layers = Vec::with_capacity(n);
                   for i in 0..n {
                       layers.push(Linear::new(
                           layer_sizes[i],
                           layer_sizes[i+1],
                           activations[i],
                           dropout_rates[i],
                           sampling_rates[i],
                       ));
                   }
                   Self { layers, task }
               }


               pub fn softmax(logits: &[f64]) -> Vec<f64> {
                   let max = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                   let exp: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
                   let sum: f64 = exp.iter().sum();
                   exp.iter().map(|&e| e / sum).collect()
               }

               pub fn sigmoid(x: f64) -> f64 {
                   if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) }
               }

               pub fn forward(&mut self, inputs: &[f64], training: bool) -> Vec<f64> {
                   let mut current = inputs.to_vec();
                   for layer in &mut self.layers {
                       current = layer.forward(&current, training);
                   }
                   current
               }

               pub fn backward(&mut self, d_outputs: &mut [f64]) -> (Vec<f64>, Vec<Vec<(Vec<f64>, f64)>>) {
                   let mut d_current = d_outputs.to_vec();
                   let mut all_grads = Vec::with_capacity(self.layers.len());
                   for layer in self.layers.iter_mut().rev() {
                       let (d_in, grads) = layer.backward(&mut d_current, false);
                       all_grads.push(grads);
                       d_current = d_in;
                   }
                   all_grads.reverse();
                   (d_current, all_grads)
               }

               pub fn fit(
                   &mut self,
                   X: &[Vec<f64>],
                   y: &[f64],
                   epochs: usize,
                   learning_rate: f64,
                   batch_size: usize,
                   optimizer_type: &str,
                   verbose: bool,
               ) {
                   let mut optimizer: Box<dyn Optimizer> = match optimizer_type {
                       "sgd" => Box::new(SGD::new(learning_rate)),
                       "adam" => Box::new(Adam::new(learning_rate, 0.9, 0.999, 1e-8)),
                       _ => panic!("Unsupported optimizer"),
                   };
                   let n_samples = X.len();
                   let mut rng = rng();

                   for epoch in 0..epochs {
                       let mut total_loss = 0.0;
                       let mut correct = 0;

                       let mut indices: Vec<usize> = (0..n_samples).collect();
                       indices.shuffle(&mut rng);

                       for start in (0..n_samples).step_by(batch_size) {
                           let end = (start + batch_size).min(n_samples);
                           let batch_indices = &indices[start..end];
                           let batch_size_actual = batch_indices.len();

                           let mut accum_grads: Vec<Vec<(Vec<f64>, f64)>> = self.layers.iter()
                           .map(|layer| layer.neurons.iter()
                           .map(|n| (vec![0.0; n.weights.len()], 0.0))
                           .collect())
                           .collect();

                           let mut batch_loss = 0.0;
                           let mut batch_correct = 0;

                           for &idx in batch_indices {
                               let inputs = &X[idx];
                               let target = y[idx];

                               let outputs = self.forward(inputs, true);

                               let (loss, d_output) = match self.task {
                                   Task::Binary => {
                                       let logit = outputs[0];
                                       let prob = Self::sigmoid(logit);
                                       let loss = 0.5 * (prob - target).powi(2);
                                       let grad = (prob - target) * prob * (1.0 - prob);
                                       if (prob >= 0.5) as i32 == target as i32 { batch_correct += 1; }
                                       (loss, vec![grad])
                                   }
                                   Task::Multi => {
                                       let target_idx = target as usize;
                                       let probs = Self::softmax(&outputs);
                                       let loss = -probs[target_idx].max(1e-12).ln();
                                       let mut grad = probs.clone();
                                       grad[target_idx] -= 1.0;
                                       let pred = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                                       if pred == target_idx { batch_correct += 1; }
                                       (loss, grad)
                                   }
                                   Task::Regression => {
                                       let pred = outputs[0];
                                       let loss = 0.5 * (pred - target).powi(2);
                                       let grad = pred - target;
                                       (loss, vec![grad])
                                   }
                               };
                               batch_loss += loss;

                               let mut d_current = d_output;
                               let mut layer_idx = self.layers.len();
                               let num_layers = layer_idx;

                               for layer in self.layers.iter_mut().rev() {
                                   layer_idx -= 1;
                                   let (d_in, grads) = layer.backward(&mut d_current, layer_idx == num_layers - 1);
                                   for (neuron_idx, (grad_w, grad_b)) in grads.into_iter().enumerate() {
                                       let (acc_w, acc_b) = &mut accum_grads[layer_idx][neuron_idx];
                                       for i in 0..acc_w.len() {
                                           acc_w[i] += grad_w[i];
                                       }
                                       *acc_b += grad_b;
                                   }
                                   d_current = d_in;
                               }
                           }

                           for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                               for (neuron_idx, neuron) in layer.neurons.iter_mut().enumerate() {
                                   if neuron.frozen { continue; }
                                   let (acc_w, acc_b) = &accum_grads[layer_idx][neuron_idx];
                                   let avg_w: Vec<f64> = acc_w.iter().map(|&g| g / batch_size_actual as f64).collect();
                                   let avg_b = *acc_b / batch_size_actual as f64;
                                   let id = format!("layer_{}_neuron_{}", layer_idx, neuron_idx);
                                   optimizer.update(neuron, &avg_w, avg_b, &id);
                               }
                           }

                           total_loss += batch_loss;
                           correct += batch_correct;
                       }

                       if verbose && (epoch % 10 == 0 || epoch == epochs - 1) {
                           let avg_loss = total_loss / n_samples as f64;
                           let accuracy = correct as f64 / n_samples as f64;
                           println!("Epoch {}, Loss: {:.6}, Accuracy: {:.4}", epoch, avg_loss, accuracy);
                       }
                   }
               }

               pub fn predict(&mut self, X: &[Vec<f64>]) -> Vec<f64> {
                   let mut preds = Vec::with_capacity(X.len());
                   for inputs in X {
                       let outputs = self.forward(inputs, false);
                       let pred = match self.task {
                           Task::Binary => {
                               let prob = Self::sigmoid(outputs[0]);
                               if prob >= 0.5 { 1.0 } else { 0.0 }
                           }
                           Task::Multi => {
                               let probs = Self::softmax(&outputs);
                               probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as f64
                           }
                           Task::Regression => outputs[0],
                       };
                       preds.push(pred);
                   }
                   preds
               }

               pub fn predict_proba(&mut self, X: &[Vec<f64>]) -> Vec<Vec<f64>> {
                   if self.task == Task::Regression {
                       panic!("predict_proba not supported for regression");
                   }
                   let mut probs = Vec::new();
                   for inputs in X {
                       let outputs = self.forward(inputs, false);
                       if self.task == Task::Binary {
                           let p = Self::sigmoid(outputs[0]);
                           probs.push(vec![1.0 - p, p]);
                       } else {
                           probs.push(Self::softmax(&outputs));
                       }
                   }
                   probs
               }

               pub fn freeze_layer(&mut self, idx: usize) {
                   if idx < self.layers.len() { self.layers[idx].freeze_all(); }
               }
               pub fn unfreeze_layer(&mut self, idx: usize) {
                   if idx < self.layers.len() { self.layers[idx].unfreeze_all(); }
               }
               pub fn freeze_neuron(&mut self, layer_idx: usize, neuron_idx: usize) {
                   if layer_idx < self.layers.len() {
                       self.layers[layer_idx].freeze_neuron(neuron_idx);
                   }
               }
               pub fn unfreeze_neuron(&mut self, layer_idx: usize, neuron_idx: usize) {
                   if layer_idx < self.layers.len() {
                       self.layers[layer_idx].unfreeze_neuron(neuron_idx);
                   }
               }
}
