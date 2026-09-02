use std::collections::HashMap;
use crate::nn::neuron::Neuron;
use crate::optimizer::Optimizer;

/// Adam optimizer state per neuron.
#[derive(Clone)]
struct AdamState {
    m_w: Vec<f64>,
    v_w: Vec<f64>,
    m_b: f64,
    v_b: f64,
    t: usize,
}

impl AdamState {
    fn new(weights_len: usize) -> Self {
        Self {
            m_w: vec![0.0; weights_len],
            v_w: vec![0.0; weights_len],
            m_b: 0.0,
            v_b: 0.0,
            t: 0,
        }
    }
}

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    states: HashMap<String, AdamState>,
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            states: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, neuron: &mut Neuron, grad_weights: &[f64], grad_bias: f64, neuron_id: &str) {
        // Retrieve or create state for this neuron
        let state = self.states.entry(neuron_id.to_string()).or_insert_with(|| {
            AdamState::new(neuron.weights.len())
        });

        // Increment timestep
        state.t += 1;
        let t = state.t as f64;

        // Update first and second moment estimates for weights
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.epsilon;
        let lr = self.learning_rate;

        for i in 0..neuron.weights.len() {
            let g = grad_weights[i];

            // m_w = beta1 * m_w + (1 - beta1) * g
            state.m_w[i] = beta1 * state.m_w[i] + (1.0 - beta1) * g;
            // v_w = beta2 * v_w + (1 - beta2) * g^2
            state.v_w[i] = beta2 * state.v_w[i] + (1.0 - beta2) * g * g;

            // Bias correction
            let m_hat = state.m_w[i] / (1.0 - beta1.powf(t));
            let v_hat = state.v_w[i] / (1.0 - beta2.powf(t));

            // Update weight
            neuron.weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        // Update first and second moment estimates for bias
        let g_b = grad_bias;
        state.m_b = beta1 * state.m_b + (1.0 - beta1) * g_b;
        state.v_b = beta2 * state.v_b + (1.0 - beta2) * g_b * g_b;

        let m_hat_b = state.m_b / (1.0 - beta1.powf(t));
        let v_hat_b = state.v_b / (1.0 - beta2.powf(t));

        neuron.bias -= lr * m_hat_b / (v_hat_b.sqrt() + eps);
    }
}
