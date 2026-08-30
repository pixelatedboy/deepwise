use rand::Rng;

mod fanctional;
use crate::fanctional::Activation;

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation: Activation,
    pub frozen: bool,

    // buffer
    last_input: Option<Vec<f64>>,
    last_output: Option<f64>,
    grad_weights_buffer: Vec<f64>,
    d_inputs_buffer: Vec<f64>,
}

impl Neuron {
    pub fn new(n_inputs: usize, activation: Activation) -> Self {
        let limit = match activation {
            Activation::Tanh => (6.0 / (n_inputs as f64 + 1.0)).sqrt(),
            Activation::Relu => (2.0 / n_inputs as f64).sqrt(),
            _ => 0.5,
        }

        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..n_inputs)
            .map(|_| rng.gen_range(-limit...limit))
            .collect()
        Neuron {
            weights,
            bias: 0.0,
            activation,
            frozen: false,
            last_input: None,
            last_output: None,
            grad_weights_buffer: vec![0.0; n_inputs],
            d_inputs_buffer: vec![0.0; n_inputs],
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> f64 {
        self.last_input = Some(inputs.to_vec());
        let z = self.bias + self.weights.iter().zip(inputs).map(|(w, x)| w * x).sum::<f64>();
        let out = self.activation.apply(z);
        self.last_output = Some(out);
        out
    }

    pub fn backward(&mut self, d_output: f64, _is_last_layer: bool) -> (Vec<f64>, f64, Vec<f64>) {
        let out = self.last_output.expect("forward must be called before backward");
        let delta = d_output * self.activation.derivative(out);
        let inputs = self.last_input.as_ref().expect("no input saved");

        for i in 0..self.weights.len() {
            self.grad_weights_buffer[i] = delta * inputs[i];
            self.d_inputs_buffer[i] = delta * self.weights[i];
        }
        let grad_bias = delta;

        (self.grad_weights_buffer.clone(), grad_bias, self.d_inputs_buffer.clone())
    }
}
