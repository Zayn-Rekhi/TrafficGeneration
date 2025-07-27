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
            "Sizeloss": nn.MSELoss(reduction='sum'),
            "Velloss": nn.MSELoss(reduction='sum'),
            "Directionloss": nn.MSELoss(reduction='sum'),
            "IOUloss": nn.MSELoss(reduction='sum'),
            "Laneloss": nn.CrossEntropyLoss(reduction='sum'),
            "Actloss": nn.CrossEntropyLoss(reduction='sum'),
        }

        self.use_logger = self.opts['use_logger']
        
        if self.use_logger:
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

            if epoch == self.opts["freeze_encoder_epoch"]:
                print(f"FREEZING ENCODER AT EPOCH: {epoch}")
                self.model.freeze_encoder()

            self.train_step(metrics_recorder, epoch)
            self.validation_step(metrics_recorder, epoch)

            metrics = metrics_recorder.average()

            if epoch == 0:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

                print(f"Total number of parameters: {total_params}")
                print(f"Total number of trainable parameters: {trainable_params}")


            if self.use_logger and epoch % self.opts["save_freq"] == 0:
                best_metrics = metrics_recorder.update_best()
                print(best_metrics)

                if "Loss_Val" in best_metrics:
                    self.logger.save_model(self.model, f"best_val_loss.pth")

                if "Posloss_Val" in best_metrics:
                    self.logger.save_model(self.model, f"best_val_pos_loss.pth")
            
            if self.use_logger:
                self.logger.log_metrics(metrics, step=epoch)
    
            print(f"Epoch {epoch}: {metrics}")

        if self.use_logger:
            self.logger.close()

    def train_step(self, metrics_recorder, step):
        self.model.train()

        for _, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            pos, size, vel, acttype, direction, laneidx, log_var, mu = self.model(data)
                
            pos_loss = self.loss_dict['Posloss'](pos, data.pos)
            size_loss = self.loss_dict['Sizeloss'](size, data.dimen)
            vel_loss = self.loss_dict['Velloss'](vel, data.vel)
            act_loss = self.loss_dict['Actloss'](acttype, data.actor_type)
            direc_loss = self.loss_dict['Directionloss'](direction, data.direction)
            lane_loss = self.loss_dict['Laneloss'](laneidx, data.lane_index)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            
            loss = 0

            if step > self.opts['include_pos_at_epoch']:
                loss += self.opts['pos_weight'] * pos_loss
            
            if step > self.opts['include_size_at_epoch']:
                loss += self.opts['size_weight'] * size_loss
            
            if step > self.opts['include_vel_at_epoch']:
                loss += self.opts['vel_weight'] * vel_loss
            
            if step > self.opts['include_actor_at_epoch']:
                loss += self.opts['actor_weight'] * act_loss
            
            if step > self.opts['include_direction_at_epoch']:
                loss += self.opts['direction_weight'] * direc_loss
            
            if step > self.opts['include_lane_at_epoch']:
                loss += self.opts['lane_weight'] * lane_loss
            
            if step > self.opts['include_kld_at_epoch']:
                loss += self.opts['kld_weight'] * kld_loss
                

            loss.backward()

            self.optimizer.step()
            # self.scheduler.step()

            metrics_recorder.record(
                {
                    "Posloss_Train": pos_loss.item(),
                    "Sizeloss_Train": size_loss.item(),
                    "Velloss_Train": vel_loss.item(),
                    "Actloss_Train": act_loss.item(),
                    "Direcloss_Train": direc_loss.item(),
                    "Laneloss_Train": lane_loss.item(),
                    "KLDloss_Train": kld_loss.item(),
                    "Loss_Train": loss.item(),
                    "Mu_Mean": mu.mean().item(),
                    "Mu_Std": mu.std().item(),
                    "Log_Var_Mean": log_var.mean().item(),
                    "Log_Var_Std": log_var.std().item(),
                    "Log_Var_Exp_Mean": log_var.exp().mean().item(),
                    #"Valimages": plot_comparison(data.pos[0], pos[0], data.edge_index[0], idx),
                }
            )
        

    def validation_step(self, metrics_recorder, step):
        with torch.no_grad():
            self.model.eval()

            for idx, data in enumerate(self.validation_loader):
                data = data.to(self.device)

                pos, size, vel, acttype, direction, laneidx, log_var, mu = self.model(data)
                
                pos_loss = self.loss_dict['Posloss'](pos, data.pos)
                size_loss = self.loss_dict['Sizeloss'](size, data.dimen)
                vel_loss = self.loss_dict['Velloss'](vel, data.vel)
                act_loss = self.loss_dict['Actloss'](acttype, data.actor_type)
                direc_loss = self.loss_dict['Directionloss'](direction, data.direction)
                lane_loss = self.loss_dict['Laneloss'](laneidx, data.lane_index)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

                loss = (
                    self.opts['pos_weight'] * pos_loss +
                    self.opts['size_weight'] * size_loss +
                    self.opts['vel_weight'] * vel_loss + 
                    self.opts['actor_weight'] * act_loss + 
                    self.opts['direction_weight'] * direc_loss + 
                    self.opts['lane_weight'] * lane_loss + 
                    self.opts['kld_weight'] * kld_loss
                )

                metrics_recorder.record(
                    {
                        "Posloss_Val": pos_loss.item(),
                        "Sizeloss_Val": size_loss.item(),
                        "Velloss_Val": vel_loss.item(),
                        "Actloss_Val": act_loss.item(),
                        "Direcloss_Val": direc_loss.item(),
                        "Laneloss_Val": lane_loss.item(),
                        "KLDloss_Val": kld_loss.item(),
                        "Loss_Val": loss.item(),
                    }
                )



if __name__ == "__main__":
    random.seed(42) # make sure every time has the same training and validation sets
    opts = read_train_yaml(os.getcwd(), filename = "experiment.yaml")
    runner = Runner(opts)
    runner.train()
    
