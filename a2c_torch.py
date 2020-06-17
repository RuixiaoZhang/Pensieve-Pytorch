import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
S_INFO = 4

def get_entropy(step):
        if step < 5000:
            return 4
        elif step < 10000:
            return 3
        elif step < 20000:
            return 2
        elif step< 30000:
            return 1
        else:
            return np.clip(1.0 - (step-30000)/50000, 0.1, 1.0)

def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H       


class ActorNet(nn.Module):
    def __init__(self, s_dim, a_dim, lr):
        super(ActorNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr_rate = lr
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(1, 128)
        self.conv1 = nn.Conv1d(1, 128, kernel_size=4)
        self.conv2 = nn.Conv1d(1, 128, kernel_size=4)
        self.conv3 = nn.Conv1d(1, 128, kernel_size=4)
        self.fc3 = nn.Linear(1, 128)
        self.out_linear = nn.Linear(2048, A_DIM)
        


    def forward(self, inputs):
        # print(inputs[:, 4:5, :A_DIM])
        # print(inputs[:, 4:5, :A_DIM].size())
        split_0 = F.relu(self.fc1(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1(inputs[:, 2:3, :].view(-1, 1, self.s_dim[1])))
        # split_2 = F.relu(self.conv1(inputs[:, 2:3, :]))
        split_3 = F.relu(self.conv2(inputs[:, 3:4, :].view(-1, 1, self.s_dim[1])))
        # split_3 = F.relu(self.conv2(inputs[:, 3:4, :]))
        split_4 = F.relu(self.conv3(inputs[:, 4:5, :A_DIM].view(-1, 1, A_DIM)))
        split_5 = F.relu(self.fc3(inputs[:, 5:6, -1]))

        # split_2_flatten, split_3_flatten, split_4_flatten = nn.Flatten(split_2, start_dim=1), nn.Flatten(split_3, start_dim=1), nn.Flatten(split_4, start_dim=1)
        split_2_flatten, split_3_flatten, split_4_flatten = split_2.flatten(start_dim=1), split_3.flatten(start_dim=1), split_4.flatten(start_dim=1)
        # print(split_2_flatten.size(), split_3_flatten.size())
        merge_net = torch.cat([split_0, split_1, split_2_flatten, split_3_flatten, split_4_flatten, split_5], dim=1)
        logits = self.out_linear(merge_net)
        
        return logits


    def get_actor_out(self, inputs):
        logits = self.forward(inputs)
        probs = F.softmax(logits, dim=1)
        # print(probs.detach())
        return logits, probs, probs.detach()
    
    def cal_loss(self, s_batch, a_batch, td_batch, epoch):
        entropy_weight = get_entropy(epoch)
        _, probs, _ = self.get_actor_out(s_batch)
        a_entropy = probs * torch.log(probs)
        loss = torch.sum(-td_batch * a_batch * torch.log(probs) + entropy_weight * a_entropy)
        # print(loss, torch.mean(loss))
        return loss

class CriticNet(nn.Module):
    def __init__(self, s_dim, lr):
        super(CriticNet, self).__init__()
        self.s_dim = s_dim
        self.lr_rate = lr
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(1, 128)
        self.conv1 = nn.Conv1d(1, 128, 4)
        self.conv2 = nn.Conv1d(1, 128, 4)
        self.conv3 = nn.Conv1d(1, 128, 4)
        self.fc3 = nn.Linear(1, 128)
        self.out_linear = nn.Linear(2048, 1)
        


    def forward(self, inputs):
        split_0 = F.relu(self.fc1(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1(inputs[:, 2:3, :]))
        split_3 = F.relu(self.conv2(inputs[:, 3:4, :]))
        split_4 = F.relu(self.conv3(inputs[:, 4:5, :A_DIM]))
        split_5 = F.relu(self.fc3(inputs[:, 5:6, -1]))

        # split_2_flatten, split_3_flatten, split_4_flatten = nn.Flatten(split_2), nn.Flatten(split_3), nn.Flatten(split_4)
        split_2_flatten, split_3_flatten, split_4_flatten = split_2.flatten(start_dim=1), split_3.flatten(start_dim=1), split_4.flatten(start_dim=1)
        
        merge_net = torch.cat([split_0, split_1, split_2_flatten, split_3_flatten, split_4_flatten, split_5], dim=1)
        out = self.out_linear(merge_net)
        return out


    def get_value_out(self, inputs):
        value = self.forward(inputs)
        return value, value.detach()
    

    def cal_loss(self, s_batch, r_batch, terminal):
        ba_size = len(s_batch)
        assert len(s_batch) == len(r_batch)
        v_out, v_batch = self.get_value_out(s_batch)
        R_batch = torch.from_numpy(np.zeros([ba_size, 1])).double()

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        td_batch = R_batch - v_out
        loss = torch.mean(td_batch.pow(2), axis=0)
        
        return loss, td_batch.detach()
