import os
import pickle
import networkx as nx
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
from scipy import stats
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from Spatial_Encoder import TheoryGridCellSpatialRelationEncoder
from PIL import Image
import torchvision.transforms as transforms
import math


def __add_gaussain_noise(img, scale):
    ow, oh = img.size
    mean = 12
    sigma = 4
    gauss = np.random.normal(mean,sigma,(ow, oh))
    gauss = gauss.reshape(ow, oh)
    img = np.array(img, dtype = np.uint8)

    mask_0 = (img == 0)
    mask_255 = (img == 255)

    img[mask_0] = img[mask_0] + gauss[mask_0]
    img[mask_255] = img[mask_255] - gauss[mask_255]
    return img


def get_transform(noise_range = 0.0, noise_type = None, isaug = False, rescale_size = 64):
    transform_list = []
    # transform_list.append(transforms.Resize(rescale_size)) 

    # if isaug:
    #     transform_list.append(transforms.RandomResizedCrop((rescale_size,rescale_size)))
    #     transform_list.append(transforms.RandomHorizontalFlip())
    #     transform_list.append(transforms.RandomRotation(45))
        # transform_list.append(transforms.GaussianBlur(3, sigma=(0, 3.0)))

    if noise_type == 'gaussian':
        transform_list.append(transforms.Lambda(lambda img: __add_gaussain_noise(img, noise_range)))

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5), (0.25))]
    return transforms.Compose(transform_list)



class Block():
    def __init__(self, data_id, block_max_dim=None, block_shape_latent = None, bldg_graph = None, mid_axis=None):
        self.data_id = data_id
        self.block_max_dim = block_max_dim
        self.block_shape_latent = block_shape_latent
        self.bldg_graph = bldg_graph
        self.mid_axis = mid_axis

    def get_attribut_value(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            print('No corresponding "{}" attribute found. '.format(attr))
            return None    



GRAPH_EXTENSIONS = [
    '.gpickle',
]

PROCESSED_EXTENSIONS = [
    '.gpickle',
]

### global defintion
quantile_level = 9
node_attr_onehot_classnum = [2, 10, 10, 11] # node_type, posx, posy, bldg_area
edge_attr_onehot_classnum = [10, 4] #edge_dist, edge_type


frequency_num = 32
f_act = 'relu'
max_radius = 10000
min_radius = 10
freq_init = 'geometric'


spa_enc = TheoryGridCellSpatialRelationEncoder(
    64,
    coord_dim = 2,
    frequency_num = frequency_num,
    max_radius = max_radius,
    min_radius = min_radius,
    freq_init = freq_init)



idx_enc = TheoryGridCellSpatialRelationEncoder(
    64,
    coord_dim = 2,
    frequency_num = frequency_num,
    max_radius = max_radius,
    min_radius = min_radius,
    freq_init = freq_init)




def is_graph_file(filename): 
    return any(filename.endswith(extension) for extension in GRAPH_EXTENSIONS)

def is_processed_file(filename):
    return any(filename.endswith(extension) for extension in PROCESSED_EXTENSIONS)


def get_node_attribute(g, keys, dtype, default = None):
    attri = list(nx.get_node_attributes(g, keys).items())
    attri = np.array(attri)
    attri = attri[:,1]
    attri = np.array(attri, dtype = dtype)
    return attri


def get_edge_attribute(g, keys, dtype, default = None):
    attri = list(nx.get_edge_attributes(g, keys).items())
    attri = np.array(attri)
    attri = attri[:,1]
    attri = np.array(attri, dtype = dtype)
    return attri



def graph2vector_processed(g):
    num_nodes = g.number_of_nodes()
    # num_edges = g.number_of_edges()

    asp_rto = g.graph['aspect_ratio']
    longside = g.graph['long_side']

    posx = get_node_attribute(g, 'posx', np.double)
    posy = get_node_attribute(g, 'posy', np.double)

    size_x = get_node_attribute(g, 'size_x', np.double)
    size_y = get_node_attribute(g, 'size_y', np.double)

    exist = get_node_attribute(g, 'exist', np.int_)
    merge = get_node_attribute(g, 'merge', np.int_)

    b_shape = get_node_attribute(g, 'shape', np.int_)
    b_iou = get_node_attribute(g, 'iou', np.double)

    node_attr = np.stack((exist, merge), 1)

    edge_list = np.array(list(g.edges()), dtype=np.int_)
    edge_list = np.transpose(edge_list)

    node_pos = np.stack((posx, posy), 1)
    node_size = np.stack((size_x, size_y), 1)

    node_idx = np.stack((np.arange(num_nodes) / num_nodes, np.arange(num_nodes) / num_nodes), axis = 1)

    return node_size, node_pos, node_attr, edge_list, node_idx, asp_rto, longside, b_shape, b_iou




# def transform_to_quantile(arr, level, except_idx = None):
#     # res = np.vectorize(lambda x: stats.percentileofscore(arr, x))(arr)
#     rang = 1.0 / np.float32(level)
#     if except_idx != None:
#         bins = np.nanquantile(arr[except_idx:], np.arange(rang,1+rang,rang))
#         arr[except_idx:] = np.digitize(arr[except_idx:], bins)
#         res = arr
#     else:
#         bins = np.nanquantile(arr, np.arange(rang,1+rang,rang))    
#         res = np.digitize(arr, bins)
#     return res

def test_graph_transform(data):
    num_nodes = data.x.shape[0]
    node_feature = data.x
    # edge_attr = data.edge_attr
    edge_idx = data.edge_index

    org_node_size = data.node_size
    org_node_size = torch.tensor(org_node_size, dtype=torch.float32)

    node_size = spa_enc(np.expand_dims(org_node_size, axis=0))
    node_size = node_size.squeeze(0)


    org_node_pos = data.node_pos
    org_node_pos = torch.tensor(org_node_pos, dtype=torch.float32)

    node_pos = spa_enc(np.expand_dims(org_node_pos, axis=0))
    node_pos = node_pos.squeeze(0)

    b_shape_gt = torch.tensor(data.b_shape, dtype=torch.int64)
    b_shape = torch.tensor(F.one_hot(b_shape_gt.clone().detach(), num_classes = 6), dtype=torch.float32)
    b_iou = torch.tensor(data.b_iou, dtype=torch.float32).unsqueeze(1)

    node_feature = torch.tensor(node_feature, dtype=torch.float32)

    # edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    edge_idx = torch.tensor(edge_idx, dtype=torch.long)

    # linkage = data.linkage
    # linkage = torch.tensor(linkage, dtype=torch.long)
    
    node_idx = torch.tensor(data.node_idx, dtype=torch.float32)
    node_idx = idx_enc(np.expand_dims(data.node_idx, axis=0))
    node_idx = node_idx.squeeze(0)

    long_side =  torch.tensor(data.long_side, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)
    asp_rto =  torch.tensor(data.asp_rto, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    long_side_gt = torch.tensor(data.long_side, dtype=torch.float32)
    asp_rto_gt =  torch.tensor(data.asp_rto, dtype=torch.float32)

    blockshape_latent =  torch.tensor(data.blockshape_latent / 10.0, dtype=torch.float32).repeat(num_nodes, 1)
    block_scale =  torch.tensor(data.block_scale / 100.0, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    blockshape_latent_gt = torch.tensor(data.blockshape_latent / 10.0, dtype=torch.float32)
    block_scale_gt =  torch.tensor(data.block_scale / 100.0, dtype=torch.float32)
    # print('shape: ',blockshape_latent.shape, blockshape_latent_gt.shape, block_scale.shape, block_scale_gt.shape, asp_rto.shape, asp_rto_gt.shape)
    # print('org: ', node_feature.shape)
    # print(torch.mean(data.blockshape_latent), data.blockshape_latent.shape)

    # print(node_feature.shape, blockshape_latent.shape, block_scale.shape, asp_rto.shape)


    # print(data.block_condition.shape)

    trans_data = Data(x=node_feature, edge_index = edge_idx, node_pos = node_pos, org_node_pos = org_node_pos, node_size = node_size, org_node_size = org_node_size, 
                        node_idx = node_idx, asp_rto = asp_rto, long_side = long_side, asp_rto_gt = asp_rto_gt, long_side_gt = long_side_gt, 
                        b_shape = b_shape, b_iou = b_iou, b_shape_gt = b_shape_gt,
                        blockshape_latent = blockshape_latent, blockshape_latent_gt = blockshape_latent_gt, block_scale = block_scale, block_scale_gt = block_scale_gt,
                        block_condition = data.block_condition, org_binary_mask = data.org_binary_mask)

    return trans_data




def graph_transform(data):
    num_nodes = data.x.shape[0]
    node_feature = data.x
    # edge_attr = data.edge_attr
    edge_idx = data.edge_index

        
    org_node_size = data.node_size
    org_node_pos = data.node_pos
    b_shape_org = data.b_shape
    b_iou = data.b_iou
    
    # print('org: ', org_node_pos)
    if torch.rand(1) < 0.5:
        org_node_size = np.flip(org_node_size, 0).copy()
        org_node_pos = np.flip(org_node_pos, 0).copy()
        b_shape_org = np.flip(b_shape_org, 0).copy()
        b_iou = np.flip(b_iou, 0).copy()
        node_feature = np.flip(node_feature, 0).copy()
    # print('flipped org: ', org_node_pos)

    # print(org_node_size.shape,org_node_pos.shape, b_shape_org.shape, b_iou.shape, node_feature.shape)

    org_node_size = torch.tensor(org_node_size, dtype=torch.float32)

    node_size = spa_enc(np.expand_dims(org_node_size, axis=0))
    node_size = node_size.squeeze(0)

    org_node_pos = torch.tensor(org_node_pos, dtype=torch.float32)

    node_pos = spa_enc(np.expand_dims(org_node_pos, axis=0))
    node_pos = node_pos.squeeze(0)

    b_shape_gt = torch.tensor(b_shape_org, dtype=torch.int64)
    b_shape = torch.tensor(F.one_hot(b_shape_gt.clone().detach(), num_classes = 6), dtype=torch.float32)
    b_iou = torch.tensor(data.b_iou, dtype=torch.float32).unsqueeze(1)



    node_feature = torch.tensor(node_feature, dtype=torch.float32)

    # edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    edge_idx = torch.tensor(edge_idx, dtype=torch.long)

    # linkage = data.linkage
    # linkage = torch.tensor(linkage, dtype=torch.long)
    
    node_idx = torch.tensor(data.node_idx, dtype=torch.float32)
    node_idx = idx_enc(np.expand_dims(data.node_idx, axis=0))
    node_idx = node_idx.squeeze(0)

    long_side =  torch.tensor(data.long_side, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)
    asp_rto =  torch.tensor(data.asp_rto, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    long_side_gt = torch.tensor(data.long_side, dtype=torch.float32)
    asp_rto_gt =  torch.tensor(data.asp_rto, dtype=torch.float32)

    blockshape_latent =  torch.tensor(data.blockshape_latent / 10.0, dtype=torch.float32).repeat(num_nodes, 1)
    block_scale =  torch.tensor(data.block_scale / 100.0, dtype=torch.float32).repeat(num_nodes).unsqueeze(1)

    blockshape_latent_gt = torch.tensor(data.blockshape_latent / 10.0, dtype=torch.float32)
    block_scale_gt =  torch.tensor(data.block_scale / 100.0, dtype=torch.float32)
    # print('shape: ',blockshape_latent.shape, blockshape_latent_gt.shape, block_scale.shape, block_scale_gt.shape, asp_rto.shape, asp_rto_gt.shape)
    # print('org: ', node_feature.shape)
    # print(torch.mean(data.blockshape_latent), data.blockshape_latent.shape)

    # print(node_feature.shape, blockshape_latent.shape, block_scale.shape, asp_rto.shape)


    # print(data.block_condition.shape)

    trans_data = Data(x=node_feature, edge_index = edge_idx, node_pos = node_pos, org_node_pos = org_node_pos, node_size = node_size, org_node_size = org_node_size, 
                        node_idx = node_idx, asp_rto = asp_rto, long_side = long_side, asp_rto_gt = asp_rto_gt, long_side_gt = long_side_gt, 
                        b_shape = b_shape, b_iou = b_iou, b_shape_gt = b_shape_gt,
                        blockshape_latent = blockshape_latent, blockshape_latent_gt = blockshape_latent_gt, block_scale = block_scale, block_scale_gt = block_scale_gt,
                        block_condition = data.block_condition, org_binary_mask = data.org_binary_mask)

    return trans_data

    # self.transform() to do augmentation




class UrbanGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, is_teaser_set = False, cnn_transform = None):
        super().__init__(root, transform, pre_transform)
        self.is_teaser_set = is_teaser_set
        self.root_data_path = root
        self.block_scale_dict = pickle.load(open(os.path.join(self.root_data_path, 'blockscale'), 'rb'))
        self.block_shape_dict = pickle.load(open(os.path.join(self.root_data_path,'blockshape_latent'), 'rb'))
        self.cnn_transforms = cnn_transform
        self.base_transform = transforms.Compose([transforms.ToTensor()])

    @property
    def raw_file_names(self):
        raw_graph_dir = []
        for root, _, fnames in sorted(os.walk(self.raw_dir)):
            for fname in fnames:
                if is_graph_file(fname):
                    path = os.path.join(root, fname)
                    raw_graph_dir.append(path)          
        return raw_graph_dir


    @property
    def processed_file_names(self):
        processed_graph_dir = []
        for root, _, fnames in sorted(os.walk(self.processed_dir)):
            for fname in fnames:
                if is_processed_file(fname):
                    path = os.path.join(root, fname)
                    processed_graph_dir.append(path)          
        return processed_graph_dir


    def process(self):
        print('processed')
        # self.dataset_attribute_discretize() # self.root inherited from BaseDataset


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # called by get_item() in BaseDatset, after get(), base dataset will implement transform()
        # if not self.is_teaser_set:
        tmp_graph = nx.read_gpickle(os.path.join(self.processed_dir, '{}.gpickle'.format(idx)))
        block_mask = Image.open(os.path.join(self.root_data_path,'binary_mask', '{}.png'.format(idx)))
        blockshape_latent = self.block_shape_dict[idx]
        block_scale = self.block_scale_dict[idx]
        # else:
        #     fn = self.processed_file_names[idx]
        #     tmp_graph = nx.read_gpickle(fn)
        #     file_id = fn[fn.rfind('/')+1:-8]
        #     block_mask = Image.open(os.path.join(self.root_data_path,'binary_mask', '{}.png'.format(file_id)))
        #     blockshape_latent = self.block_shape_dict[file_id]
        #     block_scale = self.block_scale_dict[file_id]            



        trans_image = self.cnn_transforms(block_mask)
        scale_channel = math.log(block_scale, 20) / 2.0
        if scale_channel < 0.0:
            scale_channel = 0.0
        scale_channel = torch.tensor([scale_channel]).expand(trans_image.shape)
        # print(scale_channel.shape, trans_image.shape)
        block_condition = torch.cat((trans_image, scale_channel), 0)
        # org_image = self.base_transform(org_image)
        # print(block_condition.shape)

        node_size, node_pos, node_feature, edge_index, node_idx, asp_rto, long_side, b_shape, b_iou = graph2vector_processed(tmp_graph)
        data = Data(x = node_feature, node_pos = node_pos, edge_index=edge_index, node_size = node_size, node_idx = node_idx, asp_rto = asp_rto, long_side=long_side, b_shape=b_shape, b_iou=b_iou, 
                        blockshape_latent = blockshape_latent, block_scale = block_scale, block_condition = block_condition,
                        org_binary_mask = self.base_transform(block_mask))
        return data




# root = os.getcwd()
# a = UrbanGraphDataset(os.path.join(root,'dataset','synthetic'),transform = graph_transform)
# train_loader = DataLoader(a, batch_size=6, shuffle=True)
# count = 0
# for data in train_loader:
#     # print(data)
#     # print(count)
#     print(data.x, data.edge_index, data.node_pos)
#     count += 1