import os
import sys
import logging
import wandb

import torch  # Uncomment if you want to save PyTorch models


class Logger:
    def __init__(
        self,
        log_dir,
        wandb_project=None,
        wandb_name=None,
        wandb_entity=None,
        wandb_config=None,
        wandb_isoffline=False,
    ):
        """
        Initialize the logger with paths for TensorBoard and wandb. Also sets up file logging.

        :param log_dir: Directory where TensorBoard and log.txt will be saved.
        :param wandb_project: The Weights & Biases project name (optional). If None, wandb is not used.
        :param wandb_name: The run name for Weights & Biases (optional).
        :param wandb_config: Configuration for the wandb run (optional).
        :param wandb_isoffline: Whether to run Weights & Biases offline (optional). Default is False.
        """
        self.log_dir = log_dir

        # Setup wandb if wandb_project is provided
        self.wandb_run = None
        if wandb_project is not None:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                config=wandb_config,
                mode="offline" if wandb_isoffline else None,
            )
            self.wandb_run = wandb.run

        

    def _setup_file_logging(self, log_file):
        """Sets up the logging system to log terminal output to both console and a file."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),  # Stream to console
            ],
        )

    def log_metrics(self, metrics, step):
        """
        Logs metrics to both TensorBoard and Weights & Biases.

        :param metrics: Dictionary of metric names and values.
        :param step: Current step or epoch.
        """
    
        wandb.log(metrics, step=step)

    def log_imgs(self, imgs, step):
        wandb.log({f"Images {step}": [wandb.Image(img) for img in imgs]})

    def save_model(self, model, model_name="model.pth"):
        """
        Save a PyTorch model to a file and also log to Weights & Biases if available.

        :param model: The PyTorch model to save.
        :param model_name: The name of the file to save the model to.
        """
        model_path = os.path.join(self.log_dir, model_name)
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

        if self.wandb_run:
            wandb.save(model_path)

    def close(self):
        """Close the TensorBoard writer and finalize wandb if used."""
        if self.wandb_run:
            wandb.finish()


class MetricRecorder:
    """
    MetricRecorder is a utility class for recording and averaging metric terms over multiple iterations in an epoch.
    Methods:
        __init__:
            Initializes the MetricRecorder with an empty dictionary to store metrics.
        record(d_metrics):
            Records metric terms. Each argument should be a keyword (name of the metric term)
        average_metric():
            Calculates the average metric for each term recorded over the epoch.
            Returns a dictionary with each metric term's name as the key and its average as the value.
        reset():
            Resets the recorded metrics, typically called at the start of a new epoch.
    """
    def __init__(self):
        self._metrics = {}
        self._best_metrics = {}

    def record(self, d_metrics):
        """
        Record metric terms. Each argument should be a keyword (name of the metric term)
        and the corresponding metric value (float).
        """
        for name, value in d_metrics.items():
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(value)

    def average(self):
        """
        Calculate the average metric for each term recorded over the epoch.
        Returns:
            A dictionary with each metric term's name as the key and its average as the value.
        """
        avg_metrics = {}
        for name, values in self._metrics.items():
            avg_metrics[name] = sum(values) / len(values) if values else 0.0
        return avg_metrics

    def update_best(self):
        """
        Update the best metrics with the current metrics if they are better.
        """
        improved_metrics = []

        avg_metrics = self.average()
        for name in self._metrics:
            if name not in self._best_metrics:
                self._best_metrics[name] = avg_metrics[name]
                improved_metrics.append(name)
            else:
                if self.average()[name] > self._best_metrics[name]:
                    self._best_metrics[name] = avg_metrics[name]
                    improved_metrics.append(name)

        return improved_metrics

    def reset(self):
        """
        Reset the recorded metrics, typically called at the start of a new epoch.
        """
        for name in self._metrics:
            self._metrics[name] = []