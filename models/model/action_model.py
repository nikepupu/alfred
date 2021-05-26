

import numpy as np
import torch
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import OrderedDict
from typing import Tuple, Optional, Sequence
import os
import h5py
import numpy as np

import torch
from torch.utils import data
from torch import mode, nn

import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import Resize
from transformers import RobertaTokenizer
import pickle

ROBERTA_DIM = 1024

class ThorTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, path):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.data_root = path
        self.dirs = os.listdir(path)
        
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        with open('/home/nikepupu/create_dataset/actions.pickle', 'rb') as handle:
            self.actions = pickle.load(handle)
        # Build table for conversion between linear idx -> episode/step idx

        self.cnt = 0
        self.file_index = {}
        self.desc_index = {}
        for i in range(len(self.dirs)):
           path = os.path.join(self.data_root, str(i))
           with open(path+"/high_instruction.pickle", 'rb') as handle:
                task_desc = pickle.load(handle)
                if task_desc:      
                    for j in range(len(task_desc)):
                        self.file_index[self.cnt+j] = i
                        self.desc_index[self.cnt+j] = j

                    self.cnt += len(task_desc)

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        resize = Resize((50,50))
         
        path = os.path.join(self.data_root, str(self.file_index[idx]))
        pre_path = os.path.join(path, 'pre.jpeg')
        post_path = os.path.join(path, 'post.jpeg')
        pre_image = (read_image(pre_path,))
        post_image = (read_image(post_path))

        pre_image = pre_image / 255.0
        post_image = post_image / 255.0
        
        action_file_path = os.path.join(path, 'action.txt')
        with open(action_file_path, 'r') as f:
            a = str(f.read())
        
        with open(path+"/low_instruction.pickle", 'rb') as handle:
            low_descs = pickle.load( handle)
            
        with open(path+"/high_instruction.pickle", 'rb') as handle:
            task_desc = pickle.load(handle)
        try:    
          goal_encoding = self.tokenizer(task_desc[self.desc_index[idx]])['input_ids']
          instruction_encoding = self.tokenizer(low_descs[self.desc_index[idx]] )['input_ids']
        except:
          goal_encoding = torch.zeros(10)
          insturction_encoding = torch.zeros(10)
        try:
            action_index = self.actions.index(a)
        except:
            action_index = 0

        return pre_image, action_index, post_image, instruction_encoding, goal_encoding 
    
def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in range(len(array_dict)):
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y):
    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    # print("one_hot indices", indices)
    # print("ont_hot indices shape", indices.size())
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result

class ContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).
    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, hinge=1., sigma=0.5, encoder='large',
                 ignore_action=False, copy_action=False):
        super(ContrastiveSWM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        
        
        self.lstm_low = nn.LSTM(768,ROBERTA_DIM//2, 2 ,batch_first=True, bidirectional=True)
        self.lstm_task = nn.LSTM(768,ROBERTA_DIM//2, 2 ,batch_first=True, bidirectional=True)
        
        self.reward_predictor = nn.Linear(hidden_dim, 1)
       
        
        self.pos_loss = 0
        self.neg_loss = 0
        
        num_channels = input_dims[0]
        width_height = input_dims[1:]
       

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects,
            
            )

        self.transition_model = TransitionGNN(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action)

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()
    

    def contrastive_loss(self, obs, action, next_obs):
       
        objs = self.obj_extractor(obs)
        next_objs = self.obj_extractor(next_obs)
        
        state = self.obj_encoder(objs)
        next_state = self.obj_encoder(next_objs)
        # Sample negative state across episodes at random
        
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]

        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)
        
        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state, no_trans=True)).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_objects,
                 ignore_action=False, copy_action=False, act_fn='relu'):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states, action):

        cuda = states.is_cuda
        
        batch_size = states.size(0)
        num_nodes = states.size(1)
        # print("states.shape ", states.shape)
        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)
        # print("node_attr.shape ", node_attr.shape)
        # print("num_nodes", num_nodes)
        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)

            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col], edge_attr)

        if not self.ignore_action:

            if self.copy_action:
                action_vec = to_one_hot(
                    action, self.action_dim).repeat(1, self.num_objects)
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                # print("self.action_dim", self.action_dim)
                # print("num_nodes", num_nodes)
                # print(action)
                action_vec = to_one_hot(
                    action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)

            # Attach action to each state
            # print(action_vec.shape)
            # print(node_attr.shape)
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(batch_size, num_nodes, -1)


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (2, 2), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = get_act_fn(act_fn_hid)
        self.act2 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))
    
    
class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='leaky_relu'):
        super(EncoderCNNMedium, self).__init__()

        self.cnn1 = nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim
     
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)

    def forward(self, ins):

        h_flat = ins.view(-1, self.num_objects, self.input_dim)
       
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class DecoderMLP(nn.Module):
    """MLP decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim + num_objects, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, np.prod(output_size))

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.output_size = output_size

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)

    def forward(self, ins):
        obj_ids = torch.arange(self.num_objects)
        obj_ids = to_one_hot(obj_ids, self.num_objects).unsqueeze(0)
        obj_ids = obj_ids.repeat((ins.size(0), 1, 1)).to(ins.get_device())

        h = torch.cat((ins, obj_ids), -1)
        h = self.act1(self.fc1(h))
        h = self.act2(self.fc2(h))
        h = self.fc3(h).sum(1)
        return h.view(-1, self.output_size[0], self.output_size[1],
                      self.output_size[2])


class DecoderCNNSmall(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNSmall, self).__init__()

        width, height = output_size[1] // 10, output_size[2] // 10

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=10, stride=10)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.deconv1(h_conv))
        return self.deconv2(h)


class DecoderCNNMedium(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNMedium, self).__init__()

        width, height = output_size[1] // 5, output_size[2] // 5

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=5, stride=5)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=9, padding=4)

        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        return self.deconv2(h)


class DecoderCNNLarge(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNLarge, self).__init__()

        width, height = output_size[1], output_size[2]

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=3, padding=1)

        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.ln2 = nn.BatchNorm2d(hidden_dim)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)
        self.act4 = get_act_fn(act_fn)
        self.act5 = get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        h = self.act4(self.ln1(self.deconv2(h)))
        h = self.act5(self.ln1(self.deconv3(h)))
        return self.deconv4(h)


class LookAheadPolicy(nn.Module):
    def __init__(self, input_dimm, hidden_dim, output_dim):
        super(LookAheadPolicy, self).__init__()
        self.input_dimm = input_dimm
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dimm, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dimm)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        return x
    
    
class ModelFreePolicy(nn.Module):
    def __init__(self, num_outputs) -> None:
        super(ModelFreePolicy, self).__init__()
        self.num_inputs_per_agent = 3
        final_cnn_channels = 128
        state_repr_length = 512
        self.num_outputs = num_outputs
        self.num_agents = 1
        self.cnn = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.num_inputs_per_agent, 16, 5, stride=1, padding=2
                        ),
                    ),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv2", nn.Conv2d(16, 16, 5, stride=1, padding=1)),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv3", nn.Conv2d(16, 32, 4, stride=1, padding=1)),
                    ("maxpool3", nn.MaxPool2d(3, 3)),
                    ("relu3", nn.ReLU(inplace=True)),
                    # shape =
                    ("conv4", nn.Conv2d(32, 64, 3, stride=1, padding=1)),
                    ("maxpool4", nn.MaxPool2d(3, 3)),
                    ("relu4", nn.ReLU(inplace=True)),
                    # shape = (4, 4)
                    (
                        "conv5",
                        nn.Conv2d(64, final_cnn_channels, 3, stride=1, padding=1),
                    ),
                    ("maxpool5", nn.MaxPool2d(3, 3)),
                    ("relu5", nn.ReLU(inplace=True)),
                    # shape = (2, 2)
                ]
            )
        )
        # LSTM
        self.lstm = nn.LSTM(
            final_cnn_channels * 4,
            state_repr_length,
            batch_first=True,
        )

        # Linear actor
            
        self.actor_linear = nn.Linear(state_repr_length, self.num_outputs)
            

        if self.actor_linear is not None:
            self.actor_linear.weight.data = norm_col_init(
                self.actor_linear.weight.data, 0.01
            )
            self.actor_linear.bias.data.fill_(0)
        else:
            for al in self.actor_linear_list:
                al.weight.data = norm_col_init(al.weight.data, 0.01)
                al.bias.data.fill_(0)

        # Linear critic
       
        self.critic_linear = nn.Linear(state_repr_length, 1)

        # Setting initial weights
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.cnn._modules["conv1"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv2"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv3"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv4"].weight.data.mul_(relu_gain)
        self.cnn._modules["conv5"].weight.data.mul_(relu_gain)

        
        self.train()

    def forward(
        self,
        inputs: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor],
        agent_rotations: Sequence[int],
    ):

        if inputs.shape != (self.num_agents, self.num_inputs_per_agent, 300, 300):
            raise Exception("input to model is not as expected, check!")
        
        x = self.cnn(inputs)
        # x.shape == (2, 128, 2, 2)
        
        #x = x.view(x.size(0), -1)
        x = torch.reshape(x, (x.size(0), -1))
        # x.shape = [num_agents, 512]

        # x = torch.cat((x, self.agent_num_embeddings), dim=1)
        # # x.shape = [num_agents, 512 + agent_num_embed_length]

        x, hidden = self.lstm(x.unsqueeze(1), hidden)
        # x.shape = [num_agents, 1, state_repr_length]
        # hidden[0].shape == [1, num_agents, state_repr_length]
        # hidden[1].shape == [1, num_agents, state_repr_length]

        x = x.squeeze(1)
        # x.shape = [num_agents, state_repr_length]

      
       
        value_all = self.critic_linear(x)

        to_return = {
            "critic": value_all,
            "hidden": hidden,
          
        }
  
     
        logits = self.actor_linear(x)
           
            # logits = self.actor_linear(
            #     state_talk_reply_repr
            # ) + self.marginal_linear_actor(state_talk_reply_repr).unsqueeze(1).repeat(
            #     1, self.coordinate_actions_dim, 1
            # ).view(
            #     self.num_agents, self.num_outputs ** 2
            # )
        to_return["actor"] = logits
    
        
        return to_return
    
    
if __name__ == "__main__":
    model = ModelFreePolicy(13)
    dataset = ThorTransitionsDataset('/home/nikepupu/create_dataset/dataset_train')
    # controller = Controller()
    # controller.start(player_screen_height=300,
    #              player_screen_width=300)
    
    # controller.reset(scene_name =  'floorPlan2')
    # frame = torch.Tensor(controller.last_event.frame.copy()).permute(2, 0, 1).unsqueeze(0)
    def collate_fn(batch):
        pre_image, action, post_image, low_descs, task_desc = zip(*batch)
    
        low_descs = [torch.LongTensor(i) for i in low_descs]
        task_desc = [torch.LongTensor(i) for i in task_desc]
        
        low_descs = pad_sequence(low_descs, batch_first=True, padding_value=1)
        task_desc = pad_sequence(task_desc, batch_first=True, padding_value=1)
        from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
        from transformers import RobertaConfig
        configuration = RobertaConfig().from_pretrained("roberta-base")

        roberta = RobertaEmbeddings(configuration)
    
        return torch.stack(pre_image, 0), torch.tensor(action), torch.stack(post_image, 0), roberta(low_descs), roberta(task_desc)
    
    train_loader = data.DataLoader( dataset, batch_size=2, collate_fn=collate_fn, shuffle=True, num_workers=0)
    obs = train_loader.__iter__().next()[0]
    input_shape = obs[0].size()
    
    hidden1 = torch.zeros(1, 1, 512)
    hidden2 = torch.zeros(1, 1, 512)
    hidden = tuple((hidden1, hidden2))
    
    # res = model(frame, hidden, None)
    
    world_model = ContrastiveSWM(
        embedding_dim=2,
        hidden_dim= 512,
        action_dim=13,
        input_dims= input_shape,
        num_objects=20,
        sigma=0.5,
        hinge=1,
        ignore_action=False,
        copy_action=False,
        encoder='large'
    )

    ob, action, next_obs, low_descs, task_desc = train_loader.__iter__().next()
    # print(low_descs.shape)
    # print(task_desc.shape)
    print("action, ", action)
    loss = world_model.contrastive_loss(ob, action, next_obs)