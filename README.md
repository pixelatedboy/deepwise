# Deepwise

Deepwise is a lightweight neural network library written in Rust with Python bindings via PyO3. It provides a flexible, extensible framework for building and training neural networks with features like dropout, neuron freezing, and custom architectures.

## Features

- Pure Rust Core: High-performance neural network implementation in Rust
- Python Bindings: Seamless integration with Python through PyO3
- Flexible Architecture: Build custom networks with inheritance and method overriding
- Layer Types: Fully connected (Linear) layers with multiple activations
- Training Support: Mini-batch training with SGD and Adam optimizers
- Regularization: Dropout and neuron sampling for better generalization
- Neuron Freezing: Freeze specific neurons or entire layers for transfer learning
- Multiple Tasks: Binary classification, multi-class classification, and regression
- Serialization: Save and load trained models

## Usage

### From Source (with Cargo)

```bash
git clone https://github.com/pixelatedboy/deepwise.git
cd deepwise

cargo build --release

cp target/release/libdeepwise.so target/release/deepwise_rs.so

export PYTHONPATH=$(pwd)/target/release:$PYTHONPATH
```

### Using Maturin (Recommended)

```bash
pip install .
```

## Docs

Coming Soon!
