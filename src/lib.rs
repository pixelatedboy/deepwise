use pyo3::prelude::*;

mod functional;
mod nn;
mod optimizer;
mod norm;
mod saveload;

use functional::activation::Activation;
use nn::linear::Linear;
use nn::network::{Network, Task};
use saveload::{save, load};
use norm::{Normalizer, SimpleNormalizer};

#[pyclass(name = "Linear")]
pub struct PyLinear {
    inner: Linear,
}

#[pymethods]
impl PyLinear {
    #[new]
    pub fn new(
        num_inputs: usize,
        num_neurons: usize,
        activation: &str,
        dropout_rate: f64,
        sampling_rate: f64,
    ) -> PyResult<Self> {
        let activation_enum = match activation {
            "tanh" => Activation::Tanh,
            "relu" => Activation::Relu,
            "sigmoid" => Activation::Sigmoid,
            "linear" => Activation::Linear,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("invalid activation")),
        };
        Ok(PyLinear {
            inner: Linear::new(num_inputs, num_neurons, activation_enum, dropout_rate, sampling_rate),
        })
    }

    pub fn forward(&mut self, inputs: Vec<f64>, training: bool) -> Vec<f64> {
        self.inner.forward(&inputs, training)
    }

    pub fn backward(&mut self, d_outputs: Vec<f64>, is_last_layer: bool) -> (Vec<f64>, Vec<(Vec<f64>, f64)>) {
        let mut d = d_outputs;
        self.inner.backward(&mut d, is_last_layer)
    }

    pub fn freeze_neuron(&mut self, idx: usize) {
        self.inner.freeze_neuron(idx);
    }
    pub fn unfreeze_neuron(&mut self, idx: usize) {
        self.inner.unfreeze_neuron(idx);
    }
    pub fn freeze_all(&mut self) {
        self.inner.freeze_all();
    }
    pub fn unfreeze_all(&mut self) {
        self.inner.unfreeze_all();
    }
    pub fn get_frozen_mask(&self) -> Vec<bool> {
        self.inner.get_frozen_mask()
    }

    pub fn num_neurons(&self) -> usize {
        self.inner.neurons.len()
    }

    pub fn get_weights(&self, neuron_idx: usize) -> PyResult<Vec<f64>> {
        self.inner.neurons.get(neuron_idx)
        .map(|n| n.weights.clone())
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("neuron_idx out of range"))
    }

    pub fn set_weights(&mut self, neuron_idx: usize, weights: Vec<f64>) -> PyResult<()> {
        let neuron = self.inner.neurons.get_mut(neuron_idx)
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("neuron_idx out of range"))?;
        if weights.len() != neuron.weights.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("weights length does not match neuron's input size"));
        }
        neuron.weights = weights;
        Ok(())
    }

    pub fn get_bias(&self, neuron_idx: usize) -> PyResult<f64> {
        self.inner.neurons.get(neuron_idx)
        .map(|n| n.bias)
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("neuron_idx out of range"))
    }

    pub fn set_bias(&mut self, neuron_idx: usize, bias: f64) -> PyResult<()> {
        let neuron = self.inner.neurons.get_mut(neuron_idx)
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("neuron_idx out of range"))?;
        neuron.bias = bias;
        Ok(())
    }
}

#[pyclass(name = "Network", subclass)]
pub struct PyNetwork {
    inner: Network,
}

#[pymethods]
impl PyNetwork {
    #[new]
    pub fn new() -> Self {
        PyNetwork {
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

    pub fn set_layers(&mut self, layers: Vec<Py<PyLinear>>, py: Python) -> PyResult<()> {
        let mut rust_layers = Vec::with_capacity(layers.len());
        for py_layer in &layers {
            let py_linear = py_layer.borrow(py);
            rust_layers.push(py_linear.inner.clone());
        }
        self.inner.layers = rust_layers;
        Ok(())
    }

    pub fn forward(&mut self, inputs: Vec<f64>, training: bool) -> Vec<f64> {
        self.inner.forward(&inputs, training)
    }

    pub fn backward(&mut self, d_outputs: Vec<f64>) -> (Vec<f64>, Vec<Vec<(Vec<f64>, f64)>>) {
        let mut d = d_outputs;
        self.inner.backward(&mut d)
    }

    pub fn num_layers(&self) -> usize {
        self.inner.layers.len()
    }

    #[staticmethod]
    pub fn sigmoid(x: f64) -> f64 {
        Network::sigmoid(x)
    }

    #[staticmethod]
    pub fn softmax(logits: Vec<f64>) -> Vec<f64> {
        Network::softmax(&logits)
    }

    pub fn fit(
        &mut self,
        X: Vec<Vec<f64>>,
        y: Vec<f64>,
        epochs: usize,
        learning_rate: f64,
        batch_size: usize,
        optimizer_type: String,
        verbose: bool,
    ) -> PyResult<()> {
        let n_samples = X.len();
        let mut rng = rand::rng();

        let mut optimizer: Box<dyn optimizer::Optimizer> = match optimizer_type.as_str() {
            "sgd" => Box::new(optimizer::SGD::new(learning_rate)),
            "adam" => Box::new(optimizer::Adam::new(learning_rate, 0.9, 0.999, 1e-8)),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("optimizer must be 'sgd'")),
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

                    let outputs: Vec<f64> = self.inner.forward(inputs, true);

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

                    let mut d_current: Vec<f64> = d_output;
                    let mut layer_idx = self.inner.layers.len();
                    let num_layers = layer_idx;
                    for layer in self.inner.layers.iter_mut().rev() {
                        layer_idx -= 1;
                        let (d_in, grads): (Vec<f64>, Vec<(Vec<f64>, f64)>) =
                        layer.backward(&mut d_current, layer_idx == num_layers - 1);
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
                if self.inner.task == Task::Regression {
                    println!("Epoch {}, Loss: {:.6}", epoch, avg_loss);
                } else {
                    let accuracy = correct as f64 / n_samples as f64;
                    println!("Epoch {}, Loss: {:.6}, Accuracy: {:.4}", epoch, avg_loss, accuracy);
                }
            }
        }

        Ok(())
    }

    pub fn predict(&mut self, X: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mut preds = Vec::new();
        for inputs in &X {
            let outputs: Vec<f64> = self.inner.forward(inputs, false);
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

    pub fn predict_proba(&mut self, X: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if self.inner.task == Task::Regression {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "predict_proba not supported for regression",
            ));
        }
        let mut probs = Vec::new();
        for inputs in &X {
            let outputs: Vec<f64> = self.inner.forward(inputs, false);
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

    pub fn save(&self, _filepath: &str) -> PyResult<()> {
        unimplemented!()
    }
    pub fn load(&mut self, _filepath: &str) -> PyResult<()> {
        unimplemented!()
    }
}

#[pyclass(name = "Optimizer")]
pub struct PyOptimizer {
    inner: Box<dyn optimizer::Optimizer + Send + Sync>,
}

#[pymethods]
impl PyOptimizer {
    #[staticmethod]
    pub fn sgd(learning_rate: f64) -> Self {
        PyOptimizer { inner: Box::new(optimizer::SGD::new(learning_rate)) }
    }

    #[staticmethod]
    #[pyo3(signature = (learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8))]
    pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        PyOptimizer { inner: Box::new(optimizer::Adam::new(learning_rate, beta1, beta2, epsilon)) }
    }

    pub fn update_neuron(
        &mut self,
        layer: &mut PyLinear,
        neuron_idx: usize,
        grad_weights: Vec<f64>,
        grad_bias: f64,
        neuron_id: &str,
    ) -> PyResult<()> {
        let neuron = layer.inner.neurons.get_mut(neuron_idx)
        .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("neuron_idx out of range"))?;
        self.inner.update(neuron, &grad_weights, grad_bias, neuron_id);
        Ok(())
    }

    pub fn update_layer(
        &mut self,
        layer: &mut PyLinear,
        grads: Vec<(Vec<f64>, f64)>,
                        layer_id: &str,
    ) -> PyResult<()> {
        if grads.len() != layer.inner.neurons.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "grads length does not match number of neurons in the layer",
            ));
        }
        for (neuron_idx, (grad_w, grad_b)) in grads.into_iter().enumerate() {
            let neuron = &mut layer.inner.neurons[neuron_idx];
            if neuron.frozen { continue; }
            let id = format!("{}_neuron_{}", layer_id, neuron_idx);
            self.inner.update(neuron, &grad_w, grad_b, &id);
        }
        Ok(())
    }

    pub fn update_network(
        &mut self,
        network: &mut PyNetwork,
        grads: Vec<Vec<(Vec<f64>, f64)>>,
                          network_id: &str,
    ) -> PyResult<()> {
        if grads.len() != network.inner.layers.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "grads length does not match number of layers in the network",
            ));
        }
        for (layer_idx, layer_grads) in grads.into_iter().enumerate() {
            let layer = &mut network.inner.layers[layer_idx];
            if layer_grads.len() != layer.neurons.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "grads for layer {} do not match its number of neurons", layer_idx
                )));
            }
            for (neuron_idx, (grad_w, grad_b)) in layer_grads.into_iter().enumerate() {
                let neuron = &mut layer.neurons[neuron_idx];
                if neuron.frozen { continue; }
                let id = format!("{}_layer_{}_neuron_{}", network_id, layer_idx, neuron_idx);
                self.inner.update(neuron, &grad_w, grad_b, &id);
            }
        }
        Ok(())
    }
}

#[pyclass(name = "Normalizer")]
pub struct PyNormalizer {
    inner: Normalizer,
}

#[pymethods]
impl PyNormalizer {
    #[new]
    pub fn new() -> Self {
        PyNormalizer { inner: Normalizer::new(), }
    }

    pub fn fit(&mut self, X: Vec<Vec<f64>>, y: Option<Vec<f64>>) {
        let y_ref = y.as_ref().map(|v| v.as_slice());
        self.inner.fit(&X, y_ref)
    }

    pub fn transform_x(&self, X: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.inner.transform_x(&X)
    }

    pub fn transform_y(&self, y: Vec<f64>) -> Vec<f64> {
        self.inner.transform_y(&y)
    }

    pub fn inverse_transform_y(&self, y_norm: Vec<f64>) -> Vec<f64> {
        self.inner.inverse_transform_y(&y_norm)
    }
}

#[pyclass(name = "SimpleNormalizer")]
pub struct PySimpleNormalizer {
    inner: SimpleNormalizer,
}

#[pymethods]
impl PySimpleNormalizer {
    #[new]
    pub fn new() -> Self {
        PySimpleNormalizer {
            inner: SimpleNormalizer::new(),
        }
    }

    pub fn fit(&mut self, X: Vec<Vec<f64>>) {
        self.inner.fit(&X);
    }

    pub fn transform_x(&self, X: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.inner.transform_x(&X)
    }
}

#[pymodule]
fn deepwise(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let nn_mod = PyModule::new(m.py(), "nn")?;
    nn_mod.add_class::<PyNetwork>()?;
    nn_mod.add_class::<PyLinear>()?;
    nn_mod.add_class::<PyOptimizer>()?;

    m.add_submodule(&nn_mod)?;

    let norm_mod = PyModule::new(m.py(), "norm")?;
    norm_mod.add_class::<PyNormalizer>()?;
    norm_mod.add_class::<PySimpleNormalizer>()?;

    m.add_submodule(&norm_mod);

    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
