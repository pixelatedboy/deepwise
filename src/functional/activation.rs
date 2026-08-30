use std::f64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Tanh,
    Relu,
    Linear,
    Sigmoid
}

impl Activation {
    pub fn apply(&self, z: f64) -> f64 {
        match self {
            Activation::Tanh => z.tanh(),
            Activation::Relu => if z > 0.0 { z } else { 0.0 },
            Activation::Linear => z,

            Activation::Sigmoid => {
                if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                }
            }
        }
    }

    pub fn derivative(&self, output: f64) -> f64 {
        match self {
            Activation::Tanh => 1.0 - output * output,
            Activation::Relu => if output > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => output * (1.0 - output),
            Activation::Linear => 1.0,
        }
    }
}
