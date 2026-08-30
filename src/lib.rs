use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

mod functional;
mod nn;
mod optimizer;
mod norm;
mod network;

use functional::activation::Activation;
use network::{Network, Task};

#[pyclass(name = "Network", subclass)]
pub struct PyNetwork {
    inner: Network,
}

#[pymethods]
impl PyNetwork {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Network {
                layers: Vec::new(),
                task: Task::Regression,
            }
        }
    }

    pub fn set_task(&mut self, task: &str) -> PyResult<()> {
        self.inner.task = match task {
            "binary" => Task::Binary,
            "multi" => Task::Multi,
            "regression" => Task::Regression,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("invalid task")),
        };
        Ok(())
    }

    pub fn set_layers(&mut self, layers: Vec<PyObject>) -> PyResult<()> {
        Ok(())
    }

    pub fn forward(&mut self, py: Python, inputs: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.inner.forward(&inputs, false))
    }

    pub fn fit(
        &mut self,
        py: Python,
        X: Vec<Vec<f64>>,
        y: Vec<f64>,
        epochs: usize,
        learning_rate: f64,
        batch_size: usize,
        optimizer_type: String,
        verbose: bool,
    ) -> PyResult<()> {

        let n_samples = X.len();
        let mut rng = rand::thread_rng();

        let mut optimizer: Box<dyn optimizer::Optimizer> = match optimizer_type.as_str() {
            "sgd" => Box::new(optimizer::SGD::new(learning_rate)),
            "adam" => Box::new(optimizer::Adam::new(learning_rate, 0.9, 0.999, 1e-8)),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("optimizer must be 'sgd' or 'adam'")),
        };

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;

            let mut indices: Vec<usize> = (0..n_samples).collect();
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rng);

            for start in (0..n_samples).step_by(batch_size) {
                let end = (start + batch_size).min(n_samples);
                let batch_indices = &indices[start..end];
                let batch_size_actual = batch_indices.len();

                let mut accum_grads: Vec<Vec<(Vec<f64>, f64)>> = self.inner.layers.iter()
                .map(|layer| layer.neurons.iter()
                .map(|n| (vec![0.0; n.weights.len()], 0.0))
                .collect())
                .collect();

                let mut batch_loss = 0.0;
                let mut batch_correct = 0;

                for &idx in batch_indices {
                    let inputs = &X[idx];
                    let target = y[idx];

                    let outputs: Vec<f64> = self.call_method(py, "forward", (inputs.clone(), true), None)?
                    .extract()?;

                    let (loss, d_output) = match self.inner.task {
                        Task::Binary => {
                            let logit = outputs[0];
                            let prob = Network::sigmoid(logit);
                            let loss = 0.5 * (prob - target).powi(2);
                            let grad = (prob - target) * prob * (1.0 - prob);
                            if (prob >= 0.5) as i32 == target as i32 { batch_correct += 1; }
                            (loss, vec![grad])
                        }
                        Task::Multi => {
                            let target_idx = target as usize;
                            let probs = Network::softmax(&outputs);
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
                    let mut layer_idx = self.inner.layers.len();
                    for layer in self.inner.layers.iter_mut().rev() {
                        layer_idx -= 1;
                        let (d_in, grads) = layer.backward(&mut d_current, layer_idx == self.inner.layers.len() - 1);
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

                for (layer_idx, layer) in self.inner.layers.iter_mut().enumerate() {
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

        Ok(())
    }

    pub fn predict(&mut self, py: Python, X: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mut preds = Vec::new();
        for inputs in X {
            let outputs: Vec<f64> = self.call_method(py, "forward", (inputs, false), None)?
            .extract()?;
            let pred = match self.inner.task {
                Task::Binary => {
                    let prob = Network::sigmoid(outputs[0]);
                    if prob >= 0.5 { 1.0 } else { 0.0 }
                }
                Task::Multi => {
                    let probs = Network::softmax(&outputs);
                    probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as f64
                }
                Task::Regression => outputs[0],
            };
            preds.push(pred);
        }
        Ok(preds)
    }

    pub fn predict_proba(&mut self, py: Python, X: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if self.inner.task == Task::Regression {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("predict_proba not supported for regression"));
        }
        let mut probs = Vec::new();
        for inputs in X {
            let outputs: Vec<f64> = self.call_method(py, "forward", (inputs, false), None)?
            .extract()?;
            if self.inner.task == Task::Binary {
                let p = Network::sigmoid(outputs[0]);
                probs.push(vec![1.0 - p, p]);
            } else {
                probs.push(Network::softmax(&outputs));
            }
        }
        Ok(probs)
    }

    pub fn freeze_layer(&mut self, idx: usize) {
        self.inner.freeze_layer(idx);
    }
    pub fn unfreeze_layer(&mut self, idx: usize) {
        self.inner.unfreeze_layer(idx);
    }
    pub fn freeze_neuron(&mut self, layer_idx: usize, neuron_idx: usize) {
        self.inner.freeze_neuron(layer_idx, neuron_idx);
    }
    pub fn unfreeze_neuron(&mut self, layer_idx: usize, neuron_idx: usize) {
        self.inner.unfreeze_neuron(layer_idx, neuron_idx);
    }
    pub fn freeze_all(&mut self) {
        for layer in &mut self.inner.layers { layer.freeze_all(); }
    }
    pub fn unfreeze_all(&mut self) {
        for layer in &mut self.inner.layers { layer.unfreeze_all(); }
    }
    pub fn get_frozen_status(&self) -> Vec<Vec<bool>> {
        self.inner.layers.iter().map(|l| l.get_frozen_mask()).collect()
    }
}

#[pymodule]
fn deepwise_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNetwork>()?;
    Ok(())
}
