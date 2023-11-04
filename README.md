# Multi-Level Perceptron Voting Predictor

This repository contains a Rust implementation of a simple Multi-Level Perceptron (MLP) network designed to predict voting outcomes. The MLP is built on top of the Candle deep learning framework and is trained with a dataset to predict whether the first or the second voting choice wins, or if there's a draw.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Installation

Ensure you have Rust and Candle installed on your machine. Clone the repository to your local machine:

```bash
git clone https://github.com/mm9942/voting_nn.git
cd voting_nn
```

## Usage

Compile and run the project using the following command:

```bash
cargo run
```

Follow the on-screen prompts to input voting counts and receive predictions.

## Dataset

The dataset consists of pairs of voting counts with labels indicating the outcome: 0 for the first choice winning, 1 for a draw, and 2 for the second choice winning.

## Training

The MLP is trained using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.05 over 5 epochs. Training continues until the network reaches 100% accuracy on the test set or the epoch limit is reached.

## Prediction

Input a pair of voting counts to receive a prediction on the winning side or a draw. The prediction is output to the console.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
