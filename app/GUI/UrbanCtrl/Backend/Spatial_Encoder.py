import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math


def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                    output_dim,
                    dropout_rate=None,
                    activation="sigmoid",
                    use_layernormalize=False,
                    skip_connection = False,
                    context_str = ''):
        '''

        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN

        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False
        
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        




    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.

        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output




class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                    output_dim,
                    num_hidden_layers=0,
                    dropout_rate=None,
                    hidden_dim=-1,
                    activation="sigmoid",
                    use_layernormalize=False,
                    skip_connection = False,
                    context_str = None):
        '''

        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN

        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append( SingleFeedForwardNN(input_dim = self.input_dim,
                                                    output_dim = self.output_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = False,
                                                    skip_connection = False,
                                                    context_str = self.context_str))
        else:
            self.layers.append( SingleFeedForwardNN(input_dim = self.input_dim,
                                                    output_dim = self.hidden_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = self.use_layernormalize,
                                                    skip_connection = self.skip_connection,
                                                    context_str = self.context_str))

            for i in range(self.num_hidden_layers-1):
                self.layers.append( SingleFeedForwardNN(input_dim = self.hidden_dim,
                                                    output_dim = self.hidden_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = self.use_layernormalize,
                                                    skip_connection = self.skip_connection,
                                                    context_str = self.context_str))

            self.layers.append( SingleFeedForwardNN(input_dim = self.hidden_dim,
                                                    output_dim = self.output_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = False,
                                                    skip_connection = False,
                                                    context_str = self.context_str))

        

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.

        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        return output




def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
          (frequency_num*1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = 1.0/timescales

    return freq_list


"""
The theory based Grid cell spatial relation encoder, 
See https://openreview.net/forum?id=Syx0Mh05YQ
Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion
"""
class TheoryGridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000,  min_radius = 1000, freq_init = "geometric", ffn = None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        self.unit_vec1 = np.asarray([1.0, 0.0])                        # 0
        self.unit_vec2 = np.asarray([-1.0/2.0, math.sqrt(3)/2.0])      # 120 degree
        self.unit_vec3 = np.asarray([-1.0/2.0, -math.sqrt(3)/2.0])     # 240 degree


        self.input_embed_dim = self.cal_input_dim()
        self.ffn = ffn
        
    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 6)
        self.freq_mat = np.repeat(freq_mat, 6, axis = 1)



    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)


    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        
        # (batch_size, num_context_pt, coord_dim)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec 
        # (batch_size, num_context_pt, 1)
        angle_mat1 = np.expand_dims(np.matmul(coords_mat, self.unit_vec1), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = np.expand_dims(np.matmul(coords_mat, self.unit_vec2), axis = -1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = np.expand_dims(np.matmul(coords_mat, self.unit_vec3), axis = -1)

        # (batch_size, num_context_pt, 6)
        angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis = -1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = np.expand_dims(angle_mat, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis = -2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])  # dim 2i+1
        
        return spr_embeds
    
        
    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds) 

        # if self.ffn is not None:
        #     return self.ffn(spr_embeds)
        # else:
        return spr_embeds

#把ffn和postional分开。postional当作预处理部分，ffn是encoder的一部分


def get_ffn(input_dim, f_act, context_str = ""):
    # print("Create 3 FeedForward NN!!!!!!!!!!")
    use_layn = True
    skip_connection = True
    return MultiLayerFeedForwardNN(
            input_dim=input_dim,
            output_dim=128,
            num_hidden_layers=3,
            dropout_rate=0.5,
            hidden_dim=512,
            activation=f_act,
            use_layernormalize=use_layn,
            skip_connection = skip_connection,
            context_str = context_str)



frequency_num = 32
f_act = 'relu'
max_radius = 10000
min_radius = 10
freq_init = 'geometric'

ffn = get_ffn(
    input_dim=int(6 * frequency_num),
    f_act = f_act,
    context_str = "TheoryGridCellSpatialRelationEncoder")


spa_enc = TheoryGridCellSpatialRelationEncoder(
    64,
    coord_dim = 2,
    frequency_num = frequency_num,
    max_radius = max_radius,
    min_radius = min_radius,
    freq_init = freq_init,
    ffn=ffn)


# input = np.ones([12, 40, 2], dtype=np.float)
# output = spa_enc(input)
# print(output.shape)