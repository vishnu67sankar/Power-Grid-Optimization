import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class PowerGridModelBase(nn.Module):

    def training_step(self, batch, y_mean, y_std, n_bus, loss_function=nn.MSELoss()):
        _, target = batch.x, batch.y
        y_pred = self(batch)
        batch_size = batch.num_graphs
        
        y_pred = y_pred.view(batch_size*n_bus, 2)

        y_mean_exp = y_mean.repeat(batch_size, 1)
        y_std_exp = y_std.repeat(batch_size, 1)

        y_pred = denormalize_output(y_pred, y_mean_exp, y_std_exp)
        target = denormalize_output(target, y_mean_exp, y_std_exp)

        loss = loss_function(y_pred, target)
        return loss
    
    def validation_step(self, batch, y_mean, y_std, n_bus, loss_function=nn.MSELoss()):
        _, target = batch.x, batch.y
        batch_size = batch.num_graphs

        y_pred = self(batch)
        y_pred = y_pred.view(batch_size*n_bus, 2)

        y_mean_exp = y_mean.repeat(batch_size, 1)
        y_std_exp = y_std.repeat(batch_size, 1)

        y_pred = denormalize_output(y_pred, y_mean_exp, y_std_exp)
        target = denormalize_output(target, y_mean_exp, y_std_exp)
        loss = loss_function(y_pred, target)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, history):
        batch_losses = [x['val_loss'] for x in history]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print(f"Epoch[{epoch}], train_loss: {round(result['train_loss'], 5)}, val_loss: {round(result['val_loss'], 5)}")

def denormalize_output(y_norm, y_mean, y_std):
    return (y_mean + y_norm*y_std)
