import numpy as np
import torch
from operator import is_
import pandapower as pp
import pandapower.networks as nw
# from torch_geometric.loader import DataLoader
from torch_geometric.data import DataLoader, Data

def split_dataset(dataset, split_ratio):
    """splitting data into training and validation set"""
    data_size = len(dataset)
    return dataset[:int(data_size*split_ratio)]

def make_dataset(dataset, n_bus):
    x_raw, y_raw = [], []
    for i in range(len(dataset)):
        x_sample, y_sample = [], []
        for n in range(n_bus):
            is_pv = 0
            is_pq = 0
            is_slack = 0

            if n == 0:
                is_slack = 1
            
            elif dataset[i, 4*n + 2] == 0: # Q=0, is PV bus
                is_pv = 1
            
            else:
                is_pq = 1
            
            x_sample.append([
                dataset[i, 4*n + 1],  # P
                dataset[i, 4*n + 2],  # Q
                is_pv,                # Bus type
                is_pq,
                is_slack])

            y_sample.append([dataset[i, 4*n + 3],   # V (target)
                            dataset[i, 4*n + 4]   # D (target)
                            ])

        x_raw.append(x_sample)
        y_raw.append(y_sample)

    x_raw = torch.tensor(x_raw, dtype=torch.float)
    y_raw = torch.tensor(y_raw, dtype=torch.float)
    return x_raw, y_raw

def normalize_dataset(x, y):
    x_mean, x_std = torch.mean(x, 0), torch.std(x, 0)
    y_mean, y_std = torch.mean(y, 0), torch.std(y, 0)

    x_std[x_std == 0] = 1
    y_std[y_std == 0] = 1

    x_norm = (x - x_mean)/x_std # normalize everything except bus type as it is only 1, 2, 3
    x_norm[:, :, 4] = x[:, :, 4]

    y_norm = (y - y_mean)/y_std

    return x_norm, y_norm, x_mean, y_mean, x_std, y_std

def denormalize_output(y_norm, y_mean, y_std):
    return (y_mean + y_norm*y_std)

def MSE(yhat, y):
    return torch.mean((yhat-y)**2)

def stats(train_dataset, val_dataset, train_ratio, val_ratio, n_bus):
# train_dataset = pd.read_excel('Datasets/14Bus/PF_Dataset_1.xlsx').values
# val_dataset = pd.read_excel('Datasets/14Bus/PF_Dataset_2.xlsx').values

    train_dataset = split_dataset(train_dataset, train_ratio)
    val_dataset = split_dataset(val_dataset, val_ratio)
    
    x_raw_train, y_raw_train = make_dataset(train_dataset, n_bus)
    x_raw_val, y_raw_val = make_dataset(val_dataset, n_bus)

    x_norm_train, y_norm_train, x_train_mean, y_train_mean, x_train_std, y_train_std = normalize_dataset(x_raw_train, y_raw_train)
    x_norm_val, y_norm_val, x_val_mean, y_val_mean, x_val_std, y_val_std = normalize_dataset(x_raw_val, y_raw_val)

    stats_result = [x_norm_train, y_norm_train, x_train_mean, y_train_mean, x_train_std, y_train_std, x_norm_val, y_norm_val, x_val_mean, y_val_mean, x_val_std, y_val_std]
    
    return stats_result

def main(train_dataset, val_dataset, train_ratio, val_ratio, n_bus):

    stats_result = stats(train_dataset, val_dataset, train_ratio, val_ratio, n_bus)
    x_norm_train = stats_result[0]
    y_norm_train = stats_result[1]
    x_train_mean = stats_result[2]
    y_train_mean = stats_result[3]
    x_train_std = stats_result[4]
    y_train_std = stats_result[5]
    x_norm_val = stats_result[6]
    y_norm_val = stats_result[7]
    x_val_mean = stats_result[8]
    y_val_mean = stats_result[9]
    x_val_std = stats_result[10]
    y_val_std = stats_result[11]

    if n_bus == 14:
        net = nw.case14()
    
    from_buses = net.line['from_bus'].values # 'from_bus' and 'to_bus' for each line in the network
    to_buses = net.line['to_bus'].values

    # Construct the edge index (bidirectional edges)
    edge_index = torch.tensor([list(from_buses) + list(to_buses), list(to_buses) + list(from_buses)], dtype=torch.long)

    # print("Edge Index for IEEE 14-bus System:")
    # print(edge_index)

    # Create Data objects for PyTorch Geometric
    train_data_list = [Data(x=x, y=y, edge_index=edge_index) for x, y in zip(x_norm_train, y_norm_train)]
    val_data_list = [Data(x=x, y=y, edge_index=edge_index) for x, y in zip(x_norm_val, y_norm_val)]

    # Prepare DataLoaders
    train_loader = DataLoader(train_data_list, batch_size=16)
    val_loader = DataLoader(val_data_list, batch_size=16)

    print("--------------------Data preparation completed successfully--------------------")
    return (train_loader, val_loader)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)

