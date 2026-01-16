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
from utils import read_train_yaml, plot_comparison, to_boxes

warnings.filterwarnings("ignore")



def anti_overlap_loss(pos, size, group_size):
    pos = pos.view(pos.shape[0] // group_size, group_size, pos.shape[1])
    size = size.view(size.shape[0] // group_size, group_size, size.shape[1])

    center_delta = (pos.unsqueeze(1) - pos.unsqueeze(2)).abs()
    avg_size = (size.unsqueeze(1) + size.unsqueeze(2)) / 2

    penetration = torch.clamp(center_delta - avg_size, min=0.0)
    collision = penetration[..., 0] * penetration[..., 1]
    mask = ~torch.eye(pos.size(1), device=pos.device, dtype=torch.bool)
    collision = collision[:, mask]

    return collision.sum()


def directional_cosine_loss(pred, target):
    pred = F.normalize(pred, dim=-1)  # ensure unit vectors
    target = F.normalize(target, dim=-1)
    cosine = (pred * target).sum(dim=-1)  # dot product
    angle_distance = 1 - cosine  # 1 - cos(θ)
    return angle_distance

def kl_categorical_from_logits(q_logits, p_logits=None):
    log_q = F.log_softmax(q_logits, dim=-1)
    q = log_q.exp()
    if p_logits is None:
        K = q.size(-1)
        log_p = log_q.new_full(log_q.shape, -torch.log(torch.tensor(float(K), device=log_q.device)))
    else:
        log_p = F.log_softmax(p_logits, dim=-1)
    return (q * (log_q - log_p)).sum(dim=-1)  # [B]

def kl_diag_gauss_to_gauss(mu1, logvar1, mu2, logvar2):
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    D = mu1.size(-1)
    return 0.5 * (
        (var1/var2).sum(dim=-1) +
        ((mu2 - mu1).pow(2)/var2).sum(dim=-1) -
        D + (logvar2 - logvar1).sum(dim=-1)
    )  # [...,]



class Runner:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda:{}'.format(opts['gpu_ids'][0]))

        self._setup_data()
        self._setup_model()


    def _setup_data(self):
        torch.autograd.set_detect_anomaly(True)

        dataset = TrafficDataset(self.opts['dataset_path'], self.opts['n_actors'])
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
            self.model = SceneGenerator(self.opts, self.device)
        elif self.opts['attention_generator']:
            self.model = AttentionSceneGenerator(self.opts, self.device)
        elif self.opts['attention_generator_embed']:
            self.model = AttentionSceneGeneratorWithEmbeddings(self.opts, self.device)


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
            "Directionloss": directional_cosine_loss,
            "Laneloss": nn.CrossEntropyLoss(reduction='sum'),
            "Actloss": nn.CrossEntropyLoss(reduction='sum'),
            "OverlapLoss": anti_overlap_loss,
        }

        self.use_logger = self.opts['use_wandb']
        
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

                if "Loss_Val" in best_metrics:
                    self.logger.save_model(self.model, f"best_val_loss1.pth")

                if "Posloss_Val" in best_metrics:
                    self.logger.save_model(self.model, f"best_val_pos_loss1.pth")
            
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

            pos, size, vel, acttype, direction, laneidx, log_var, mu, pi, pi_logits = self.model(data)

            pos_loss = self.loss_dict['Posloss'](pos, data.pos)
            size_loss = self.loss_dict['Sizeloss'](size, data.dimen)
            vel_loss = self.loss_dict['Velloss'](vel, data.vel)
            direc_loss = self.loss_dict['Directionloss'](direction, data.direction)
            act_loss = self.loss_dict['Actloss'](acttype, torch.argmax(data.actor_type, axis=1))
            lane_loss = self.loss_dict['Laneloss'](laneidx, torch.argmax(data.lane_index, axis=1))
            overlap_loss = self.loss_dict['OverlapLoss'](pos, size, self.opts['n_actors'])

            
            B, K, D = mu.size(0), self.model.n_components, self.model.latent_dim
            mu_kd     = mu.view(B, K, D)
            logvar_kd = log_var.view(B, K, D)

            kl_y = kl_categorical_from_logits(pi_logits, self.model.prior_logits)  # [B]

            prior_mu     = self.model.prior_mu.unsqueeze(0).expand(B, K, D)
            prior_logvar = self.model.prior_logvar.unsqueeze(0).expand(B, K, D)
            kl_k = kl_diag_gauss_to_gauss(mu_kd, logvar_kd, prior_mu, prior_logvar)  # [B,K]

            pi_probs = F.softmax(pi_logits, dim=1)  
            kl_z = (pi_probs * kl_k).sum(dim=-1) 

            kld_loss = (kl_y + kl_z).mean()

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
                loss += self.opts['direction_weight'] * direc_loss.mean()
            
            if step > self.opts['include_lane_at_epoch']:
                loss += self.opts['lane_weight'] * lane_loss
            
            if step > self.opts['include_overlap_at_epoch']:
               loss += self.opts['overlap_weight'] * overlap_loss
            
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
                    "Direcloss_Train": direc_loss.mean().item(),
                    "Laneloss_Train": lane_loss.item(),
                    "KLDloss_Train": kld_loss.item(),
                    "OverlapLoss_Train": overlap_loss.item(),
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

                pos, size, vel, acttype, direction, laneidx, log_var, mu, pi, pi_logits = self.model(data)

                pos_loss = self.loss_dict['Posloss'](pos, data.pos) 
                size_loss = self.loss_dict['Sizeloss'](size, data.dimen) 
                vel_loss = self.loss_dict['Velloss'](vel, data.vel) 
                act_loss = self.loss_dict['Actloss'](acttype, data.actor_type) 
                direc_loss = self.loss_dict['Directionloss'](direction, data.direction) 
                lane_loss = self.loss_dict['Laneloss'](laneidx, data.lane_index) 
                overlap_loss = self.loss_dict['OverlapLoss'](pos, size, self.opts['n_actors'])

                B, K, D = mu.size(0), self.model.n_components, self.model.latent_dim
                mu_kd     = mu.view(B, K, D)
                logvar_kd = log_var.view(B, K, D)

                kl_y = kl_categorical_from_logits(pi_logits, self.model.prior_logits)  # [B]

                prior_mu     = self.model.prior_mu.unsqueeze(0).expand(B, K, D)
                prior_logvar = self.model.prior_logvar.unsqueeze(0).expand(B, K, D)
                kl_k = kl_diag_gauss_to_gauss(mu_kd, logvar_kd, prior_mu, prior_logvar)  # [B,K]

                pi_probs = F.softmax(pi_logits, dim=1)  
                kl_z = (pi_probs * kl_k).sum(dim=-1) 

                kld_loss = (kl_y + kl_z).mean()

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
                    loss += self.opts['direction_weight'] * direc_loss.mean()
                
                if step > self.opts['include_lane_at_epoch']:
                    loss += self.opts['lane_weight'] * lane_loss
            
                
                if step > self.opts['include_overlap_at_epoch']:
                    loss += self.opts['overlap_weight'] * overlap_loss
                
                if step > self.opts['include_kld_at_epoch']:
                    loss += self.opts['kld_weight'] * kld_loss
                    


                metrics_recorder.record(
                    {
                        "Posloss_Val": pos_loss.item(),
                        "Sizeloss_Val": size_loss.item(),
                        "Velloss_Val": vel_loss.item(),
                        "Actloss_Val": act_loss.item(),
                        "Direcloss_Val": direc_loss.mean().item(),
                        "Laneloss_Val": lane_loss.item(),
                        "KLDloss_Val": kld_loss.item(),
                        # "IOUSloss_Val": overlap_loss.item(),
                        "Loss_Val": loss.item(),
                    }
                )



if __name__ == "__main__":
    random.seed(42) # make sure every time has the same training and validation sets
    opts = read_train_yaml(os.getcwd(), filename = "experiment.yaml")
    runner = Runner(opts)
    runner.train()