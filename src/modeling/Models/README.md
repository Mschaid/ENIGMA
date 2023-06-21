# Models for predicting neural signals 

## SimpleLSTM

This is a simple implementation of a Long Short-Term Memory (LSTM) model using TensorFlow's Keras API. The model is defined in the `SimpleLSTM.py` file and consists of a single LSTM layer followed by a dense output layer. 

### Usage

To use this model, simply import the `SimpleLSTM` class from the `SimpleLSTM.py` file and create an instance of it with the desired parameters. Then, you can train the model using your own data and evaluate its performance.

```python
from SimpleLSTM import SimpleLSTM

# Define model parameters
sequence_length = 10
num_features = 5
lstm_1_units = 64

# Create model instance
model = SimpleLSTM(sequence_length, num_features, lstm_1_units)

# Train and evaluate model
# ...
```

GitHub Copilot: Sure, here's a sample README file for the selected code:

# StackedLSTM

This is an implementation of a Stacked Long Short-Term Memory (LSTM) model using TensorFlow's Keras API. The model is defined in the `StackedLSTM.py` file and consists of two LSTM layers followed by a dense output layer. 

## Usage

To use this model, simply import the `StackedLSTM` class from the `StackedLSTM.py` file and create an instance of it with the desired parameters. Then, you can train the model using your own data and evaluate its performance.

```python
from StackedLSTM import StackedLSTM

# Define model parameters
sequence_length = 10
num_features = 5
lstm_1_units = 64

# Create model instance
model = StackedLSTM(sequence_length, num_features, lstm_1_units)

# Train and evaluate model
# ...
```

## Requirements

This code requires the following packages to be installed:

- TensorFlow
- NumPy
- Pandas

## License

This code is licensed under the MIT License. See the `LICENSE` file for more information.



