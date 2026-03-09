# Traffic Generation

An end-to-end machine learning pipeline for generating traffic data. This repository provides a modular framework for training, benchmarking, and evaluating traffic generation models.

## Repository Structure

* **`dataset.py`**: Contains data loaders, preprocessing scripts, and dataset class definitions.
* **`model.py`**: Defines the neural network architectures used for traffic generation.
* **`train.py`**: The primary training loop, handling optimization, backpropagation, and checkpointing.
* **`bench.py`**: Benchmarking script to evaluate model performance, inference speed, and generation quality.
* **`experiment.yaml`**: Configuration file to manage hyperparameters, dataset paths, and training parameters.
* **`logger.py`**: Logging utility to track training metrics such as loss and accuracy.
* **`utils.py`**: Helper functions and common utilities used across the project.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or later installed. It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Clone the repository
git clone [https://github.com/Zayn-Rekhi/TrafficGeneration.git](https://github.com/Zayn-Rekhi/TrafficGeneration.git)
cd TrafficGeneration

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies (requires a requirements.txt file)
# pip install -r requirements.txt
