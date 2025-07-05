import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.data import Batch

from logger import Logger, MetricRecorder

import os
import numpy as np
import random
import warnings
import logging
from time import gmtime, strftime

from model import *


from dataset import TrafficDataset
from utils import read_train_yaml, plot_comparison

warnings.filterwarnings("ignore")


class Runner:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda:{}'.format(opts['gpu_ids'][0]))

        self._setup_data()
        self._setup_model()


    def _setup_data(self):
        torch.autograd.set_detect_anomaly(True)

        dataset = TrafficDataset(self.opts['dataset_path'])
        num_data = len(dataset)

        val_num = int(num_data * self.opts['val_ratio']) 
        val_idx = np.array(random.sample(range(num_data), val_num))
        train_idx = np.delete(np.arange(num_data), val_idx.astype(int))

        print(f'Get {train_idx.shape[0]} graph for training')
        print(f'Get {val_idx.shape[0]} graph for validation')

        val_dataset = dataset[val_idx]
        train_dataset = dataset[train_idx]
        self.validation_loader = DataLoader(val_dataset, batch_size=self.opts['batch_size'], shuffle=False, num_workers=8)
        self.train_loader = DataLoader(train_dataset, batch_size=self.opts['batch_size'], shuffle=True, num_workers=8)

    def _setup_model(self):
        self.model = None

        if self.opts['generator']:
            self.model = BlockGenerator(self.opts, self.device)
        elif self.opts['attention_generator']:
            self.model = AttentionBlockGenerator(self.opts, self.device)

        assert self.model, "Error in model name"

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= float(self.opts['lr']), weight_decay=1e-6)
        self.scheduler = MultiStepLR(self.optimizer, 
                                     milestones=[(self.opts['total_epochs'] - self.opts['start_epoch']) * 0.6, 
                                                 (self.opts['total_epochs'] - self.opts['start_epoch']) * 0.8], 
                                     gamma=0.3)
        self.loss_dict = {
            "Posloss": nn.MSELoss(reduction='sum'),
            "Laneloss": nn.CrossEntropyLoss(reduction='sum'),
            "Actloss": nn.CrossEntropyLoss(reduction='sum'),
            "Directionloss": nn.MSELoss(reduction='sum'),
        }

        self.logger = Logger(
            log_dir=self.opts['log_dir'],
            wandb_project=self.opts['wandb_project'],
            wandb_name=self.opts['wandb_name'],
            wandb_entity=self.opts['wandb_entity'],
            wandb_config=self.opts,
        )


    def train(self):
        print('Start Training...')
        logging.info('Start Training...' )
        
        metrics_recorder = MetricRecorder()

        for epoch in range(self.opts['start_epoch'], self.opts['total_epochs']):
            metrics_recorder.reset()

            self.train_step(metrics_recorder)
            self.validation_step(metrics_recorder)

            metrics = metrics_recorder.average()
            self.logger.log_metrics(metrics, step=epoch)
            
            # if "Posloss_Val" in metrics_recorder.update_best():
            #     self.logger.save_model(self.model, f"best_pos_loss_model_{epoch}.pth")
            # else:
            #     self.logger.save_model(self.model, f"model_{epoch}.pth")

            print(f"Epoch {epoch}: {metrics}")

        self.logger.close()

    def train_step(self, metrics_recorder):
        self.model.train()

        for _, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            pos, acttype, direction, laneidx, log_var, mu = self.model(data)
                
            pos_loss = self.loss_dict['Posloss'](pos, data.pos)
            act_loss = self.loss_dict['Actloss'](acttype, data.actor_type)
            direc_loss = self.loss_dict['Directionloss'](direction, data.direction)
            lane_loss = self.loss_dict['Laneloss'](laneidx, data.lane_index)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            loss = 4 * pos_loss + 1.5 * act_loss + 1 * direc_loss + 2 * lane_loss + kld_loss

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            metrics_recorder.record(
                {
                    "Posloss_Train": pos_loss.item(),
                    "Actloss_Train": act_loss.item(),
                    "Direcloss_Train": direc_loss.item(),
                    "Laneloss_Train": lane_loss.item(),
                    "KLDloss_Train": kld_loss.item(),
                    "Loss_Train": loss.item(),
                    #"Valimages": plot_comparison(data.pos[0], pos[0], data.edge_index[0], idx),
                }
            )
        

    def validation_step(self, metrics_recorder):
        with torch.no_grad():
            self.model.eval()

            for idx, data in enumerate(self.validation_loader):
                data = data.to(self.device)

                pos, acttype, direction, laneidx, log_var, mu = self.model(data)

                
                pos_loss = self.loss_dict['Posloss'](pos, data.pos)
                act_loss = self.loss_dict['Actloss'](acttype, data.actor_type)
                direc_loss = self.loss_dict['Directionloss'](direction, data.direction)
                lane_loss = self.loss_dict['Laneloss'](laneidx, data.lane_index)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)


                loss = pos_loss + act_loss + direc_loss + lane_loss + kld_loss
 
                metrics_recorder.record(
                    {
                        "Posloss_Val": pos_loss.item(),
                        "Actloss_Val": act_loss.item(),
                        "Direcloss_Val": direc_loss.item(),
                        "Laneloss_Val": lane_loss.item(),
                        "KLDloss_Val": kld_loss.item(),
                        "Loss_Val": loss.item(),
                        #"Valimages": plot_comparison(data.pos[0], pos[0], data.edge_index[0], idx),
                    }
                )



if __name__ == "__main__":
    random.seed(42) # make sure every time has the same training and validation sets
    opts = read_train_yaml(os.getcwd(), filename = "experiment.yaml")
    runner = Runner(opts)
    runner.train()
    
