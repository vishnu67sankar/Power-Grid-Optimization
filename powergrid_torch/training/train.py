import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, val_loader, y_mean, y_std, n_bus):
    model.eval()
    outputs = [model.validation_step(batch, y_mean, y_std, n_bus) for batch in val_loader]

    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, optimizer, scheduler, y_train_mean, y_train_std, y_val_mean, y_val_std, n_bus):
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = []    

        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch, y_train_mean, y_train_std, n_bus)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
        
        result = evaluate(model, val_loader, y_val_mean, y_val_std, n_bus)
        result['train_loss'] = torch.stack(train_loss).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        scheduler.step(result['train_loss'])
        
    return history