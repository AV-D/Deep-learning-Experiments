# Neural Network Configuration Experiments

## Overview
This project is dedicated to testing various neural network configurations to determine their impact on performance. It includes experiments with different optimizers, loss functions, dropout rates, and activation functions, using a combination of TensorFlow and custom neural network implementations.

## Theoretical Background
To understand the experiments and results in this project, familiarity with the following concepts is essential:

### 1. **Neural Networks**
Neural networks are computational systems vaguely inspired by the biological neural networks of animal brains. They consist of layers of interconnected nodes or neurons, where each connection represents a weight that can be adjusted during training. Neurons in one layer take inputs from previous layers, apply a weighted sum followed by a non-linear function (activation function), and pass the output to the next layer. The basic architecture often includes:
- **Input Layer**: Accepts the features of the data.
- **Hidden Layers**: Intermediate layers where most computations are done via a weighted sum followed by an activation function.
- **Output Layer**: Produces the final output of the network, suitable for the type of problem (e.g., classification, regression).

### 2. **Optimizers**
Optimizers are algorithms or methods used to change the attributes of the neural network such as weights and learning rate to reduce the losses. Optimizers aim to minimize (or maximize) an objective function, often a loss function, that measures the difference between the actual value and the predicted value by the model. Common optimizers include:
- **Stochastic Gradient Descent (SGD)**: Updates parameters using a gradient of the loss function concerning each parameter.
- **Adam**: An algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.
- **RMSprop**: An adaptive learning rate method, designed to solve some of the problems encountered with the SGD.

### 3. **Loss Functions**
These are functions designed to quantify the difference between the true values and the values predicted by the model. They guide the optimizer by providing a measure of performance that the optimizer uses to find the best weights for the model. Common loss functions are:
- **Mean Squared Error (MSE)**: Commonly used for regression tasks.
- **Cross-Entropy Loss**: Often used in classification tasks. Measures the performance of a classification model whose output is a probability value between 0 and 1.

### 4. **Dropout**
Dropout is a regularization technique used to prevent neural networks from overfitting. During training, randomly selected neurons are ignored or "dropped out." This means their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

### 5. **Activation Functions**
Activation functions define the output of a node given an input or set of inputs. They introduce non-linear properties to the network which allows them to learn more complex patterns:
- **Sigmoid**: A function that maps any real-valued number into the (0, 1) interval, used for binary classification.
- **ReLU (Rectified Linear Unit)**: Allows models to account for interactions and non-linearities. It outputs zero for any negative input but for any positive value \( x \) it returns that value back.
- **Tanh (Hyperbolic Tangent)**: Similar to the sigmoid but outputs values between -1 and 1.

## Installation

Clone the repository using:
```
git clone https://github.com/AV-D/Deep-learning-Experiments.git
```

Navigate to the directory containing the notebook:
```
cd Deep-learning-Experiments
```

## Usage
To view and run the notebook, ensure you have Jupyter installed:
```
jupyter notebook
```
Open the notebook `Anbu_Devadasan_HW2-1.ipynb` from within Jupyter's interface.

## Experiments Conducted
- Various models with different configurations were tested to see their impact on performance metrics such as accuracy and loss.
- Specific emphasis was laid on how changes in the architecture and hyperparameters affect the learning and generalization capability of the network.

## Results
The project discusses the findings in terms of which configurations yield better performance and how significant the changes are in terms of accuracy and computational efficiency.

## References
- [Neural Network Forward Pass and Backpropagation Example](https://theneuralblog.com/forward-pass-backpropagation-example/)
- Data 255 course tutorials.

