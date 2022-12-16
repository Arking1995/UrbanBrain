"""
Backend Implementation
"""
from .Backend.model import *
from .Backend.geometry_utlis import *
import yaml
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
....
"""

def read_train_yaml(checkpoint_name, filename = "train.yaml"):
    with open(os.path.join(checkpoint_name, filename), "rb") as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return opt



class GlobalMapper:
    def __init__(self, params: dict):
        self.cur_model = None
        self.cur_latent = None
        self.cur_layout = None
        self.cur_block_condition = None
        self.cur_idx = None
        self.cur_midaxis = None
        self.cur_block = None

        self.results_dir = params['results_dir']
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        pretrained_dir = params['pretrained_dir']
        self.pretrained_dir = pretrained_dir

        gpu_ids = 0
        self.device = torch.device('cuda:' + str(gpu_ids))

        with open(os.path.join(pretrained_dir, 'z_samples'), 'rb') as f:
            self.z_samples_dict = pickle.load(f)

        with open(os.path.join(pretrained_dir, 'block_condition'), 'rb') as f:
            self.block_condition_dict = pickle.load(f)

        with open(os.path.join(pretrained_dir, 'node_edges'), 'rb') as f:
            self.edge_idx, self.node_idx = pickle.load(f)

        with open(os.path.join(pretrained_dir, 'geo_recover'), 'rb') as f:
            [self.midaxis_dict, self.block_dict, _] = pickle.load(f)


        self.idx_list = list(self.z_samples_dict.keys())
        self.cur_model, self.opt = self.initialize_model(pretrained_dir)


    def get_latent(self):
        ####### change index to random int
        # rand_idx = np.random.randint(len(self.idx_list)) 
        # self.cur_idx = self.idx_list[rand_idx]
        self.cur_idx = 35107
        self.cur_latent = self.z_samples_dict[self.cur_idx]
        self.cur_block_condition = self.block_condition_dict[self.cur_idx]
        ####### change to random latent code
        # self.cur_latent = np.expand_dims(np.random.normal(0.0, 1.0, self.opt['latent_dim']), axis = 0)  
        return self.cur_latent.copy(), self.cur_block_condition.copy()


    def get_layout(self):
        self.cur_midaxis = self.midaxis_dict[self.cur_idx]
        self.cur_block = self.block_dict[self.cur_idx]
        block = self.cur_block
        midaxis = self.cur_midaxis

        latent, block_condition = self.get_latent()
        latent = torch.from_numpy(latent).unsqueeze(0)
        block_condition = torch.from_numpy(block_condition).unsqueeze(0)
        exist, merge, posx, posy, sizex, sizey, b_shape, b_iou = self.cur_model.decode(latent.to(self.device), block_condition.to(self.device), self.edge_idx.to(self.device), self.node_idx.to(self.device))

        exist = exist.squeeze().detach().cpu().numpy()
        merge = merge.squeeze().detach().cpu().numpy()
        posx = posx.squeeze().detach().cpu().numpy()
        posy = posy.squeeze().detach().cpu().numpy()
        sizex = sizex.squeeze().detach().cpu().numpy()
        sizey = sizey.squeeze().detach().cpu().numpy()
        b_iou = b_iou.squeeze().detach().cpu().numpy()
        _, shape_pred = torch.max(b_shape, 1)
        shape_pred = shape_pred.detach().cpu().numpy()

        geo_list = [exist, merge, posx, posy, sizex, sizey, shape_pred, b_iou, block, midaxis]
        data = get_org_layout(geo_list) # return 1000*600*3 image

        return data


    def initialize_model(self, pretrained_dir: str):
        opt = read_train_yaml(pretrained_dir, 'train_save.yaml')
        opt['device'] = self.device
        model = AttentionBlockGenerator_independent_cnn(opt, N = 120)
        model.load_state_dict(torch.load(os.path.join(pretrained_dir, "latest.pth"), map_location=self.device))
        model.to(self.device)
        return model, opt




