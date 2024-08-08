import os
import pickle

import torch
from torch.autograd import Variable
import numpy as np
from model import SocialLSTM
from helper import getCoef
from grid import getSequenceGridMask, getGridMaskInference


class STGraph:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.nodes = [{} for _ in range(seq_length)]
        self.edges = [{} for _ in range(seq_length)]

    def reset(self):
        self.nodes = [{} for _ in range(self.seq_length)]
        self.edges = [{} for _ in range(self.seq_length)]

    def construct_graph(self, x_seq):
        for t in range(self.seq_length):
            frame = x_seq[t]
            for node in frame:
                ped_id = int(node[0])
                x, y = node[1], node[2]
                if ped_id not in self.nodes[t]:
                    self.nodes[t][ped_id] = (x, y)
                else:
                    self.nodes[t][ped_id] = (x, y)
                # Add temporal edges (self-loop in this context)
                if t > 0 and ped_id in self.nodes[t - 1]:
                    prev_x, prev_y = self.nodes[t - 1][ped_id]
                    self.edges[t][(ped_id, ped_id)] = ((prev_x, prev_y), (x, y))

    def get_sequence(self, idx):
        nodes = np.zeros((self.seq_length, len(self.nodes[idx]), 2))
        nodesPresent = [[] for _ in range(self.seq_length)]
        for t in range(self.seq_length):
            for i, (ped_id, pos) in enumerate(self.nodes[t].items()):
                nodes[t, i, :] = pos
                nodesPresent[t].append(i)
        return nodes, None, nodesPresent, None


class Predictor:
    def __init__(self, model_num, epoch):
        self.model_path = f'save/{model_num}/'
        self.epoch = epoch
        self.net, self.saved_args = self.load_model()

    def load_model(self):
        with open(os.path.join(self.model_path, 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)
        net = SocialLSTM(saved_args, True)
        net.to(torch.device("cpu"))

        checkpoint_path = os.path.join(self.model_path, f'social_lstm_model_{self.epoch}.tar')
        if os.path.isfile(checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path)
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print(f'Loaded checkpoint at epoch {model_epoch}')

        return net, saved_args

    def predict_trajectory(self, x_seq, obs_length, pred_length, dimensions):
        """Get the grid masks for the sequence"""
        grid_seq = getSequenceGridMask(x_seq, dimensions, self.saved_args.neighborhood_size, self.saved_args.grid_size)

        """Construct ST graph"""
        stgraph = STGraph(obs_length)
        stgraph.construct_graph(x_seq)

        """Get nodes and nodesPresent"""
        nodes, _, nodesPresent, _ = stgraph.get_sequence(0)
        nodes = Variable(torch.from_numpy(nodes).float(), requires_grad=False).to(torch.device("cpu"))

        """Extract the observed part of the trajectories"""
        obs_nodes, obs_nodesPresent, obs_grid = nodes[:obs_length], nodesPresent[:obs_length], grid_seq[:obs_length]

        """Perform trajectory prediction"""
        history_coords, pred_gaussians = self.sample(obs_nodes, obs_nodesPresent, obs_grid, obs_length, pred_length, dimensions)

        return history_coords, pred_gaussians

    def sample(self, nodes, nodes_present, grid, obs_length, pred_length, dimensions):
        num_nodes = nodes.shape[1]
        hidden_states = Variable(torch.zeros(num_nodes, self.net.args.rnn_size), requires_grad=False).to(torch.device("cpu"))
        cell_states = Variable(torch.zeros(num_nodes, self.net.args.rnn_size), requires_grad=False).to(torch.device("cpu"))

        for tstep in range(obs_length - 1):
            out_obs, hidden_states, cell_states = self.net(
                nodes[tstep].view(1, num_nodes, 2), [grid[tstep]], [nodes_present[tstep]], hidden_states, cell_states
            )

        # Initialize lists to store results
        obs_pos = []  # List for actual history coordinates [x, y]
        pred_gaussians = []  # List for predicted Gaussian parameters [mux, muy, sx, sy, corr]

        # Store actual history coordinates
        for tstep in range(obs_length):
            obs_pos.append(nodes[tstep].cpu().numpy().tolist())

        ret_nodes = Variable(torch.zeros(obs_length + pred_length, num_nodes, 2), requires_grad=False).to(torch.device("cpu"))
        ret_nodes[:obs_length, :, :] = nodes.clone()
        prev_grid = grid[-1].clone()

        for tstep in range(obs_length - 1, pred_length + obs_length - 1):
            outputs, hidden_states, cell_states = self.net(
                ret_nodes[tstep].view(1, num_nodes, 2), [prev_grid], [nodes_present[obs_length - 1]], hidden_states, cell_states
            )
            mux, muy, sx, sy, corr = getCoef(outputs)

            # Initialize list for this time step's Gaussian parameters
            gaussians_timestep = []

            for node_idx in range(num_nodes):
                gaussians_timestep.append([mux[0, node_idx].item(), muy[0, node_idx].item(), sx[0, node_idx].item(), sy[0, node_idx].item(), corr[0, node_idx].item()])

            # Append the list of this time step's Gaussian parameters to pred_gaussians
            pred_gaussians.append(gaussians_timestep)

            ret_nodes[tstep + 1, :, 0] = mux.data
            ret_nodes[tstep + 1, :, 1] = muy.data

            list_of_nodes = Variable(torch.LongTensor(nodes_present[obs_length - 1]), requires_grad=False).to(torch.device("cpu"))
            current_nodes = torch.index_select(ret_nodes[tstep + 1], 0, list_of_nodes)
            prev_grid = getGridMaskInference(current_nodes.data.cpu().numpy(), dimensions, self.saved_args.neighborhood_size, self.saved_args.grid_size)
            prev_grid = Variable(torch.from_numpy(prev_grid).float(), requires_grad=False).to(torch.device("cpu"))

        return obs_pos, pred_gaussians
