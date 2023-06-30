import torch
from torch.utils import data
import numpy as np
from encoder import *
import lava.lib.dl.slayer as slayer

class Dataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, noise=False):
        self.labels = labels.flatten()
        self.data = data.flatten()
        self.sequence_length = sequence_length
        self.all_data = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data), axis=0)
        self.all_labels = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.labels), axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            X.append(self.all_data[i])
        y = self.all_labels[idx]
        return np.array(X), np.array(y)

class SpikingDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step, min_val, max_val):
        self.labels = labels
        self.data = data[:]
        self.sequence_length = sequence_length
        self.all_data = np.concatenate((np.zeros((sequence_length, 4)), self.data), axis=0)
        self.all_labels = np.concatenate((np.zeros((sequence_length, 2)), self.labels), axis=0)
        self.min = min_val
        self.max = max_val
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        end = idx + self.sequence_length
        seq = self.all_data[idx:end, :]
        seq[seq > self.max] = self.max
        seq[seq < self.min] = self.min
        seq = (seq - self.min) / (self.max - self.min)
        seq = torch.tensor(seq)
        X = bernoulli(datum=seq, time=self.time_step, dt=1, device="cpu")
        y = self.all_labels[end - 1, :]
        X = torch.flatten(X, start_dim=1)
        X = torch.transpose(X, 0, 1)
        # X = torch.transpose(seq, 0, 1)
        return X.float(), y
    
class NewDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step, min_val, max_val):
        self.labels = labels
        self.data = data[:]
        self.sequence_length = sequence_length
        self.all_data = np.concatenate((np.zeros((sequence_length, 4)), self.data), axis=0)
        self.all_labels = np.concatenate((np.zeros((sequence_length, 2)), self.labels), axis=0)
        self.min = min_val
        self.max = max_val
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        end = idx + self.sequence_length
        seq = self.all_data[idx:end, :]
        seq[seq > self.max] = self.max
        seq[seq < self.min] = self.min
        seq = (seq - self.min) / (self.max - self.min)
        seq = torch.tensor(seq)
        y = self.all_labels[end, :]
        X = torch.transpose(seq, 0, 1)
        
        # X = slayer.utils.time.replicate(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = X.repeat_interleave(1, dim=1)
        
        return X.float(), y

class RegSpikingDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels
        self.data = data 
        self.sequence_length = sequence_length
        temp_data = np.array([[0 for _ in range(self.data.shape[1])] for _ in range(sequence_length)])
        temp_label = np.array([[0 for _ in range(self.labels.shape[1])] for _ in range(sequence_length)])
        self.all_data = np.concatenate((temp_data, self.data), axis=0)
        self.all_labels = np.concatenate((temp_label, self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_vec = []
            for j in range(self.all_data.shape[1]):
                temp = RateEncoder(self.all_data[i][j], self.min, self.max, self.time_step)
                temp = np.array(temp)
                temp_vec.append(temp)
            temp_vec = np.stack(temp_vec, axis=0)
            X.append(temp_vec)
        X = np.stack(X, axis=0)
        X = X.reshape((-1, X.shape[-1]))
        y = np.array(self.all_labels[idx])
        return X, y
    
class RegTorchSpikingDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels
        self.data = data 
        self.sequence_length = sequence_length
        temp_data = np.array([[0 for _ in range(self.data.shape[1])] for _ in range(sequence_length)])
        temp_label = np.array([[0 for _ in range(self.labels.shape[1])] for _ in range(sequence_length)])
        self.all_data = np.concatenate((temp_data, self.data), axis=0)
        self.all_labels = np.concatenate((temp_label, self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        X1 = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i])
            X1.append(torch.min(temp_tensor).item())
            
        X1 = np.min(np.array(X1), keepdims=True)
        
        temp_tensor = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i])
            temp_tensor -= X1[0]
            temp_result = bernoulli(temp_tensor, self.time_step, device="cpu")
            X.append(temp_result.detach().numpy())

        X = np.stack(X, axis=1)
        X = np.reshape(X, (X.shape[0], -1))
        X = np.transpose(X)
        
        y = np.array(self.all_labels[idx])
        
        return X, X1, y
    
class RegTorchSeasonalitySpikingDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels
        self.data = data 
        self.sequence_length = sequence_length
        temp_data = np.array([[0 for _ in range(self.data.shape[1])] for _ in range(sequence_length)])
        temp_label = np.array([[0 for _ in range(self.labels.shape[1])] for _ in range(sequence_length)])
        self.all_data = np.concatenate((temp_data, self.data), axis=0)
        self.all_labels = np.concatenate((temp_label, self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        X1 = []
        X_p = []
        
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i])
            X1.append(torch.min(temp_tensor).item())
            
        X1 = np.min(np.array(X1), keepdims=True)
        
        temp_tensor = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i][:4])
            temp_tensor -= X1[0]
            temp_result = bernoulli(temp_tensor, self.time_step, device="cpu")
            X.append(temp_result.detach().numpy())
            X_p.append(self.all_data[i][4:])

        X = np.stack(X, axis=1)
        X = np.reshape(X, (X.shape[0], -1))
        X = np.transpose(X)
        
        y = np.array(self.all_labels[idx])
        
        X_p = np.array(X_p).flatten()
        return X, X1, X_p, y
    
class RegTorchSeasonalityLinearSpikingDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step, on_loihi=False):
        self.labels = labels
        self.data = data 
        self.sequence_length = sequence_length
        temp_data = np.array([[0 for _ in range(self.data.shape[1])] for _ in range(sequence_length)])
        temp_label = np.array([[0 for _ in range(self.labels.shape[1])] for _ in range(sequence_length)])
        self.all_data = np.concatenate((temp_data, self.data), axis=0)
        self.all_labels = np.concatenate((temp_label, self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step
        self.on_loihi = on_loihi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        X1 = []
        X_p = []
        
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i])
            X1.append(torch.min(temp_tensor).item())
            
        X1 = np.min(np.array(X1), keepdims=True)
        
        temp_tensor = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i][:4])
            temp_tensor -= X1[0]
            temp_result = bernoulli(temp_tensor, self.time_step, device="cpu")
            X.append(temp_result.detach().numpy())
            X_p.append(self.all_data[i][:])

        X = np.stack(X, axis=1)
        X = np.reshape(X, (X.shape[0], -1))
        X = np.transpose(X)
        
        y = np.array(self.all_labels[idx])
        
        X_p = np.array(X_p).flatten()
        
        if self.on_loihi:
            return X, y
        else:
            return X, X1, X_p, y
        

class AMySpikingDataset(data.Dataset):
    def __init__(self, data, on_loihi=False):
        self.data = data
        self.on_loihi = on_loihi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, X1, X_p, y = self.data[idx]
        
        if self.on_loihi:
            return X, y
        else:
            return X, X1, X_p, y
    
class RegTorchBothDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels
        self.data = data 
        self.sequence_length = sequence_length
        temp_data = np.array([[0 for _ in range(self.data.shape[1])] for _ in range(sequence_length)])
        temp_label = np.array([[0 for _ in range(self.labels.shape[1])] for _ in range(sequence_length)])
        self.all_data = np.concatenate((temp_data, self.data), axis=0)
        self.all_labels = np.concatenate((temp_label, self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        X1 = []
        X2 = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i])
            X1.append(torch.min(temp_tensor).item())
            
        X1 = np.min(np.array(X1), keepdims=True)
        
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i])
            temp_tensor -= X1[0]
            temp_result = bernoulli(temp_tensor, self.time_step, device="cpu")
            X.append(temp_result.detach().numpy())
            X2.append(self.all_data[i])
        X = np.stack(X, axis=1)
        X = X.reshape((X.shape[0], -1))
        X = np.transpose(X)
        y = np.array(self.all_labels[idx])
        X1 = np.min(np.array(X1), keepdims=True)
        X2 = np.stack(X2, axis=0)
        return X, X1, X2, y
    
class RegTorchFixedSpikingDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels
        self.data = data 
        self.sequence_length = sequence_length
        temp_data = np.array([[0 for _ in range(self.data.shape[1])] for _ in range(sequence_length)])
        temp_label = np.array([[0 for _ in range(self.labels.shape[1])] for _ in range(sequence_length)])
        self.all_data = np.concatenate((temp_data, self.data), axis=0)
        self.all_labels = np.concatenate((temp_label, self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step
        self.fixed_data = []
        self.fixed_label = []
        
        for i in range(len(self.data)):
            self.fixed_data.append(self.A(i)[0])
            self.fixed_label.append(self.A(i)[1])

    def __len__(self):
        return len(self.data)

    def A(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_tensor = torch.tensor(self.all_data[i])
            temp_tensor -= torch.min(temp_tensor)
            temp_result = bernoulli(temp_tensor, self.time_step, device="cpu")
            X.append(temp_result.detach().numpy())
        X = np.stack(X, axis=1)
        X = X.reshape((X.shape[0], -1))
        X = np.transpose(X)
        y = np.array(self.all_labels[idx])
        
        return X, y
    
    def __getitem__(self, idx):
        return self.fixed_data[idx], self.fixed_label[idx]
        

class TestDataset(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    

class ARegDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels
        self.data = data 
        self.sequence_length = sequence_length
        temp_data = np.array([[0 for _ in range(self.data.shape[1])] for _ in range(sequence_length)])
        temp_label = np.array([[0 for _ in range(self.labels.shape[1])] for _ in range(sequence_length)])
        self.all_data = np.concatenate((temp_data, self.data), axis=0)
        self.all_labels = np.concatenate((temp_label, self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_vec = []
            for j in range(self.all_data.shape[1]):
                temp = self.all_data[i][j]
                temp_vec.append(temp)
            temp_vec = np.stack(temp_vec, axis=-1)
            X.append(temp_vec)
        X = np.stack(X, axis=-1)
        y = np.array(self.all_labels[idx])
        X = X.flatten()
        return X, y

class ATSRegDataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, time_step):
        self.labels = labels
        self.data = data 
        self.sequence_length = sequence_length
        temp_data = np.array([[0 for _ in range(self.data.shape[1])] for _ in range(sequence_length)])
        temp_label = np.array([[0 for _ in range(self.labels.shape[1])] for _ in range(sequence_length)])
        self.all_data = np.concatenate((temp_data, self.data), axis=0)
        self.all_labels = np.concatenate((temp_label, self.labels), axis=0)
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            temp_vec = []
            for j in range(self.all_data.shape[1]):
                temp = self.all_data[i][j]
                temp_vec.append(temp)
            temp_vec = np.stack(temp_vec, axis=-1)
            X.append(temp_vec)
        X = np.stack(X, axis=0)
        y = np.array(self.all_labels[idx])
        return X, y
    
    
    
class SpikingMNISTDataset(data.Dataset):
    def __init__(self, data, labels, time_step):
        self.all_data = data 
        self.all_labels = labels
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        # Select sample
        X = []
        for i in range(self.all_data.shape[1]):
            temp = RateEncoder(self.all_data[idx, i], self.min, self.max, self.time_step)
            X.append(temp)
        X = np.stack(X, axis=1)
        y = np.array(self.all_labels[idx])
        return X, y

class SpikingCIFAR10Dataset(data.Dataset):
    def __init__(self, data, labels, time_step):
        self.all_data = data 
        self.all_labels = labels
        self.min = np.min(self.all_data)
        self.max = np.max(self.all_data)
        self.time_step = time_step

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        # Select sample
        X = []
        for i in range(self.all_data.shape[1]):
            temp = RateEncoder(self.all_data[idx, i], self.min, self.max, self.time_step)
            X.append(temp)
        X = np.stack(X, axis=1)
        y = np.array(self.all_labels[idx])
        return X, y
