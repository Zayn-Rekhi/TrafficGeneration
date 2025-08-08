import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.data import Batch

from logger import Logger

import os
import numpy as np
import random
import yaml
import warnings
import json
import logging
from time import gmtime, strftime
import matplotlib.pyplot as plt

from model import *

from dataset import TrafficDataset
from utils import read_train_yaml, plot_comparison, plot_output, to_boxes
from sentence_transformers import SentenceTransformer
from torchvision.ops import generalized_box_iou_loss


warnings.filterwarnings("ignore")


class Benchmark:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda:{}'.format(opts['gpu_ids'][0]))
        self.loss_dict = {}

        self._setup_data()
        self._setup_model()

    def _setup_data(self):
        torch.autograd.set_detect_anomaly(True)

        if self.opts['bench_load_data']:
            dataset = TrafficDataset(self.opts['dataset_path'])
            num_data = len(dataset)

            val_num = int(num_data * self.opts['val_ratio']) 
            val_idx = np.array(random.sample(range(num_data), val_num))

            print('Get {} graph for validation'.format(val_idx.shape[0]))

            val_dataset = dataset[val_idx]

            self.validation_loader = DataLoader(val_dataset, batch_size=self.opts['bench_batch_size'], shuffle=True, num_workers=8)

        with open(self.opts['bench_config_path'], 'r') as file:
            self.config = json.load(file)

        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    def _setup_model(self):
        self.model = None

        if self.opts['generator']:
            self.model = BlockGenerator(self.opts, self.device)
        elif self.opts['attention_generator']:
            self.model = AttentionBlockGenerator(self.opts, self.device)
        elif self.opts['attention_generator_embed']:
            self.model = AttentionBlockGeneratorWithEmbeddings(self.opts, self.device)
        
        assert self.model, "Error in model name"

        self.model.load_state_dict(torch.load(self.opts['bench_path']))
        self.model.to(self.device)


    def run(self):
        with torch.no_grad():
            self.model.eval()

            for idx, data in enumerate(self.validation_loader):
                data = data.to(self.device)
                output = self.model(data)

                plot_comparison(data, output, data.edge_index,
                                config=self.config, 
                                opts=self.opts,
                                idx=idx)

    
    def test_decoder(self, num_iter):
        with torch.no_grad():
            self.model.eval()

            for idx in range(num_iter):
                sentence = ['The road structure consists of a primary bidirectional road intersected by a smaller road forming a T-junction. The primary road has two lanes in each direction, separated by a physical median, with an additional left-turn lane for one direction near the intersection. The smaller intersecting road connects from one side and merges into the primary road with a single entry lane. Dedicated bike lanes exist in both directions of the primary road, physically separated from the vehicle lanes. The intersection provides lane groups for through and left-turn movements from the primary road and a merging point from the secondary road.A car is stationary toward the an unknown direction. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is behind and to the left of a car.Independent Actor: A car is moving toward the southwest. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is ahead and to the right of a car.Independent Actor: A car is moving toward the an unknown direction. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is ahead and to the right of a car.Independent Actor: A car is moving toward the southwest. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is ahead and to the right of a car.Independent Actor: A truck_bus is moving toward the an unknown direction. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is ahead and to the right of a car.Independent Actor: A car is moving toward the an unknown direction. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is behind and to the left of a car.']
                data = torch.randn((self.opts['bench_batch_size'], self.opts['latent_dim']), device=self.device)
                edge_index = torch.zeros((2, 15))

                cnt = 0
                for idx1 in range(0, 5):
                    for idx2 in range(idx1 + 1, 6):
                        edge_index[0][cnt] = idx1
                        edge_index[1][cnt] = idx2
                        cnt += 1

                edge_index = edge_index.to(dtype=torch.int64, device=self.device)
            
                embed = torch.tensor(self.embedder.encode(sentence), dtype=torch.float32).to(self.device)
                output = self.model.decoder_only(data, edge_index, embed)


                plot_output(output, data, edge_index,
                            config=self.config, 
                            save_dir=os.path.join(self.opts['log_dir'], f"vis_gauss/plot_{idx}.jpg"))


if __name__ == "__main__":
    random.seed(42) # make sure every time has the same training and validation sets
    opts = read_train_yaml(os.getcwd(), filename = "experiment.yaml")
    bench = Benchmark(opts)
    bench.run()
    # bench.test_decoder(100)
    
