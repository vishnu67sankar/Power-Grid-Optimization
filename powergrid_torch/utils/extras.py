import numpy as np
import torch
import torchvision
import torch.nn as nn

def summary(model):
    print("--- Model Summary ---")
    print(model)
    print("\n" + "="*100 + "\n")

    print("--- Detailed Parameter Summary ---")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        print(f"{name:<50} {str(list(parameter.shape)):<20} {params}")
        # print(f"{name:<50} {str(list(parameter.shape)):<20 {params}}")
        total_params += params
    
    print(f"\n Total Trainable Params: {total_params}")
    print("\n" + "="*100 + "\n")