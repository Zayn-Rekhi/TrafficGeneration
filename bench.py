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
from itertools import combinations

from model import *

from dataset import TrafficDataset
from utils import read_train_yaml, plot_comparison, plot_output, to_boxes
from sentence_transformers import SentenceTransformer
from torchvision.ops import generalized_box_iou_loss


warnings.filterwarnings("ignore")


def make_utriangle_edge_index(n, group_size=6, device=None):
    """
    Build your older 'upper-triangle once' edge_index:
    for each graph, add one directed edge per (i<j) pair.
    Result shape for group_size=6 is [2, 15] per graph; stacked for n graphs with offsets.
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    per_graph = torch.tensor(list(combinations(range(group_size), 2)), dtype=torch.long, device=device).t()  # [2, 15]
    if n == 1:
        return per_graph

    # stack with node offsets
    edges = []
    for b in range(n):
        off = b * group_size
        edges.append(per_graph + off)
    edge_index = torch.cat(edges, dim=1)  # [2, 15*n]
    return edge_index

@torch.no_grad()
def sample_given_embeds_for_plot(model, embeds, k=None, edge_index=None, group_size=6, device=None):
    """
    Returns:
      output: (pos, size, vel, acttype, direc, laneidx)  # exactly like decoder_only
      z:      latent tensor you can pass as 'data' to plot_output
      edge_index: your old-style upper-triangle edges (or provided)
    """
    model.eval()
    device = device or next(model.parameters()).device
    embeds = embeds.to(device)
    B = embeds.size(0)

    # Condition from text (your class already has this)
    cond = model.encode_text(embeds)  # [B, latent_dim]

    # Sample z ~ prior (use learned global GMM if present, else N(0,I))
    if all(hasattr(model, a) for a in ("prior_mu", "prior_logvar", "prior_logits")):
        K, D = model.n_components, model.latent_dim
        if k is None:
            cat = torch.distributions.Categorical(logits=model.prior_logits)
            k = cat.sample((B,)).to(device)
        elif isinstance(k, int):
            k = torch.full((B,), k, dtype=torch.long, device=device)
        else:
            k = k.to(device)
        mu  = model.prior_mu[k]                                # [B, D]
        std = (0.5 * model.prior_logvar[k]).exp().clamp_min(1e-3)
        z   = mu + std * torch.randn_like(mu)                  # [B, D]
    else:
        z = torch.randn(B, model.latent_dim, device=device)    # fallback

    # Edge index (match your old API)
    if edge_index is None:
        edge_index = make_utriangle_edge_index(B, group_size=group_size, device=device)

    # Decode with condition (this mirrors your forward)
    posx, posy, sizex, sizey, vel, acttype, dcos, dsin, laneidx, dummy = model.decode(z, edge_index, condition=cond)
    pos   = torch.cat([posx, posy], dim=1)
    size  = torch.cat([sizex, sizey], dim=1)
    direc = F.normalize(torch.cat([dcos, dsin], dim=1), dim=-1)

    output = (pos, size, vel, acttype, direc, laneidx, dummy)
    return output, z, edge_index



class Benchmark:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda:{}'.format(opts['gpu_ids'][0]))
        self.loss_dict = {}

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

        



    def _setup_model(self):
        self.model = None

        if self.opts['generator']:
            self.model = BlockGenerator(self.opts, self.device)
        elif self.opts['attention_generator']:
            self.model = AttentionBlockGenerator(self.opts, self.device)
        elif self.opts['attention_generator_embed']:
            print("Initialized")
            self.model = AttentionBlockGeneratorWithEmbeddings(self.opts, self.device)
        
        assert self.model, "Error in model name"

        self.model.load_state_dict(torch.load(self.opts['bench_path']))
        self.model.to(self.device)

        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        with open(self.opts['bench_config_path'], 'r') as file:
            self.config = json.load(file)


    def run(self):
        self._setup_data()

        with torch.no_grad():
            self.model.eval()

            for idx, data in enumerate(self.validation_loader):
                data = data.to(self.device)
                output = self.model(data)

                actor_type = output[3]
                print("ACTOR TYPES:")
                print(actor_type, 
                      data.actor_type)

                # plot_comparison(data, output, data.edge_index,
                #                 config=self.config, 
                #                 opts=self.opts,
                #                 idx=idx)

    
    def test_decoder(self, num_iter):
        with torch.no_grad():
            self.model.eval()

            for idx in range(num_iter):
                # sentence = ['The road layout features a main corridor running through the scene, joined by a secondary approach that meets the corridor at a side junction. The corridor carries traffic in both directions and is divided by a planted separation, with an added turn bay tapering toward the merge area. The side approach arrives from one edge and feeds into the corridor through a single receiving channel. Protected bicycle paths parallel the corridor on each side, set apart from motor lanes by a buffer. The junction organizes movements for continuing straight or turning from the corridor, and provides a merging path from the side approach. A car is stationary facing an unspecified heading. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is behind and to the left of a car. Independent Actor: A car is moving toward the northeast. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is ahead and to the right of a car. Independent Actor: A car is moving toward an uncertain heading. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is ahead and to the right of a car. Independent Actor: A car is moving toward the southeast. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is ahead and to the right of a car. Independent Actor: A truck_bus is moving toward an uncertain heading. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is ahead and to the right of a car. Independent Actor: A car is moving toward an uncertain heading. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is behind and to the left of a car.']
                # sentence = ['Bidirectional road segment with two main lanes separated by a centerline. Diagonal striping on both sides suggests parking bays or no-parking zones. Several small side extensions branch off, resembling minor driveways or access points. A rectangular patch in the center may indicate a manhole or a calibration marker.A bicycle is stationary toward the an unknown direction. It is not located within any defined lane. Its size is not defined, possibly indicating a small or temporary presence. It is ahead and to the right of a car.Independent Actor: A car is moving toward the an unknown direction. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is behind and to the left of a bicycle.Independent Actor: A pedestrian is moving toward the an unknown direction. It is not located within any defined lane. Its size is not defined, possibly indicating a small or temporary presence. It is behind and to the left of a bicycle.Independent Actor: A car is moving toward the an unknown direction. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is behind and to the left of a bicycle.Independent Actor: A car is moving toward the an unknown direction. It is positioned within a designated traffic lane. It has a defined physical presence on the road. It is behind and to the left of a bicycle.Independent Actor: A pedestrian is moving toward the an unknown direction. It is not located within any defined lane. Its size is not defined, possibly indicating a small or temporary presence. It is behind and to the left of a bicycle.']
                sentence = ['4 exist in scenario']
                embed = torch.tensor(self.embedder.encode(sentence), dtype=torch.float32, device=self.device)
                # sample and decode (returns exactly what plot_output expects)
                output, data, edge_index = sample_given_embeds_for_plot(
                    self.model,
                    embeds=embed,                   # [B, embed_size]
                    k=9,                         # or int to force a component
                    edge_index=None,                # or pass your own
                    group_size=6,
                    device=self.device
                )

                actor_type = output[3]
                print("ACTOR TYPES:")
                print(torch.argmax(actor_type, dim=1))

                # plot_output(
                #     output, data, edge_index,
                #     target=next(iter(self.validation_loader)),
                #     config=self.config,
                #     opts=self.opts,
                #     save_dir=os.path.join(self.opts['log_dir'], f"vis_gauss/plot_{idx}.jpg")
                # )


if __name__ == "__main__":
    random.seed(42) # make sure every time has the same training and validation sets
    opts = read_train_yaml(os.getcwd(), filename = "experiment.yaml")
    bench = Benchmark(opts)
    # bench.run()
    bench.run()
    
