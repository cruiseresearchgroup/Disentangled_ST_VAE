from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .layers import *
from .utils import *

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
    
    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)


class BaseEncoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(BaseEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_dim_list[:-2], layer_dim_list[1:-1])):
            self.layers.add_module(name="Linear{:d}".format(i), 
                                module=nn.Linear(in_size, out_size))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
        
        self.linear_means = nn.Linear(layer_dim_list[-2], layer_dim_list[-1])
        self.linear_log_var = nn.Linear(layer_dim_list[-2], layer_dim_list[-1])
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        means = self.linear_means(x)
        logvar = self.linear_log_var(x)
        return means, logvar

class BaseDecoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(BaseDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_dim_list[:-2], layer_dim_list[1:-2])):
            self.layers.add_module(name="Linear{:d}".format(i), 
                                module=nn.Linear(in_size, out_size))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))

        self.layers.add_module(name="Linear{:d}".format(num_layers), 
                                module=nn.Linear(layer_dim_list[-2], layer_dim_list[-1]))
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        return x

class BaseGenerator(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(BaseGenerator, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        layers = []

        for index in range(num_layers):
            layers.append(nn.Linear(layer_dim_list[index], layer_dim_list[index+1]))
            layers.append(activation(*act_param))
        
        self.layers = nn.Sequential(*layers)
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        return x


class BaseDiscriminator(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(BaseDiscriminator, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        layers = []

        for index in range(num_layers):
            layers.append(nn.Linear(layer_dim_list[index], layer_dim_list[index+1]))
            layers.append(activation(*act_param))
        
        self.layers = nn.Sequential(*layers)
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        return x

class BaseRegressor(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(BaseRegressor, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        layers = []

        for index in range(num_layers):
            layers.append(nn.Linear(layer_dim_list[index], layer_dim_list[index+1]))
            layers.append(activation(*act_param))
        
        self.layers = nn.Sequential(*layers)
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        return x

class ConvEncoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(ConvEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvEncoder, every element except the last one 
            should be a quadruple, and the last element should be a tuple 
            contains (z_dim and the except dimension after flatten).
        '''
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        self.z_dim, self.view_size = layer_dim_list[-1]

        for i, dim_tuple in enumerate(layer_dim_list[:-1]):
            self.layers.add_module(name="Conv{:d}".format(i), 
                                    module=nn.Conv2d(*dim_tuple))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
        
        self.layers.add_module(name="Conv{:d}".format(num_layers), 
                                module=nn.Conv2d(self.view_size, 2*self.z_dim, 1))
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        means = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return means, logvar

class ConvDecoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(ConvDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvEncoder, every element except the first one 
            should be a quadruple, and the first element should be a tuple 
            contains (z_dim and the except dimension after view).
        '''
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        self.z_dim, self.view_size = layer_dim_list[0]

        self.layers.add_module(name="Conv{:d}".format(0), 
                                module=nn.Conv2d(self.z_dim, self.view_size, 1))
                                
        for i, dim_tuple in enumerate(layer_dim_list[1:]):
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
            self.layers.add_module(name="ConvTranspose2d{:d}".format(i+1), 
                                    module=nn.ConvTranspose2d(*dim_tuple))
            
        # self.weight_init()

    
    def forward(self, x):
        x = self.layers(x)
        return x

class LSTMEncoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(LSTMEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMEncoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.

            layer_dim_list format: [input_size, hidden_size, num_layers, latent_size]
        '''
        self.input_size, self.hidden_size, self.num_layers, self.latent_size  = layer_dim_list
        self.layers = nn.Sequential()

        self.layers.add_module(name="LSTM", 
                                module=nn.LSTM(
                                    input_size=self.input_size, 
                                    hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers, 
                                    batch_first=True))
        
        self.linear_means = nn.Linear(self.hidden_size, self.latent_size)
        self.linear_log_var = nn.Linear(self.hidden_size, self.latent_size)
        # self.weight_init()

    def forward(self, x):
        [batch_size, seq_len, embed_size] = x.size()

        _, (_, final_state) = self.layers(x)

        final_state = final_state.view(self.num_layers, batch_size, self.hidden_size)
        final_state = final_state[-1]

        means = self.linear_means(final_state)
        logvar = self.linear_log_var(final_state)
        return means, logvar

class LSTMDecoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(LSTMDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMDecoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.

            layer_dim_list format: [input_size, hidden_size, num_layers, seq_len]
        '''
        self.input_size, self.hidden_size, self.num_layers,  self.seq_len = layer_dim_list
        self.layers = nn.Sequential()

        self.layers.add_module(name="LSTM", 
                                module=nn.LSTM(
                                    input_size=self.input_size, 
                                    hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers, 
                                    batch_first=True))

    def forward(self, x):
        x = x.repeat(1, self.seq_len)
        x = x.view(-1, self.seq_len, self.input_size)
        x, _ = self.layers(x)
        return x

class ConvLSTMEncoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(ConvLSTMEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMEncoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.

            layer_dim_list format: [input_dim, hidden_dim, kernel_size, num_layers, image_size]
        '''
        self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, self.image_size = layer_dim_list
        self.kernel_size = tuple(self.kernel_size)
        self.layers = ConvLSTM(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                kernel_size=self.kernel_size,
                                num_layers=self.num_layers, 
                                batch_first=True)
        
        self.latent_size = self.hidden_dim[-1] * self.image_size[0] * self.image_size[1]
        self.linear_means = nn.Linear(self.latent_size, self.latent_size)
        self.linear_log_var = nn.Linear(self.latent_size, self.latent_size)


    def forward(self, x):
        [batch_size, seq_len, embed_size, height, width] = x.size()
        _, pred = self.layers(x)
        final_state = pred[0][0].view(batch_size, self.latent_size)

        means = self.linear_means(final_state)
        logvar = self.linear_log_var(final_state)
        return means, logvar

class ConvLSTMDecoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(ConvLSTMDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvLSTMDecoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.

            layer_dim_list format: [input_dim, hidden_dim, kernel_size, num_layers, seq_len, image_size]
        '''
        self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, self.seq_len, self.image_size = layer_dim_list
        self.kernel_size = tuple(self.kernel_size)

        self.layers = ConvLSTM(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                kernel_size=self.kernel_size,
                                num_layers=self.num_layers, 
                                batch_first=True)

    
    def forward(self, x):
        batch_size = x.size()[0]
        x = x.repeat(1, self.seq_len)
        x = x.view(batch_size, self.seq_len, self.input_dim, self.image_size[0], self.image_size[1])
        x, _ = self.layers(x)
        return x[0]


class RevisedConvLSTMEncoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(RevisedConvLSTMEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMEncoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.

            layer_dim_list format: [input_dim, hidden_dim, kernel_size, num_layers, convlstm_input_size]
        '''
        self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, self.convlstm_input_size = layer_dim_list[-1]
        self.kernel_size = tuple(self.kernel_size)

        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        for i, dim_tuple in enumerate(layer_dim_list[:-1]):
            self.layers.add_module(name="Conv{:d}".format(i), 
                                    module=nn.Conv2d(*dim_tuple))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))

        self.convlstm = ConvLSTM(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                kernel_size=self.kernel_size,
                                num_layers=self.num_layers, 
                                batch_first=True)
        
        self.latent_size = self.hidden_dim[-1] * self.convlstm_input_size[0] * self.convlstm_input_size[1]
        self.linear_means = nn.Linear(self.latent_size, self.latent_size)
        self.linear_log_var = nn.Linear(self.latent_size, self.latent_size)


    def forward(self, x):
        [batch_size, seq_len, embed_size, height, width] = x.size()
        x = x.view(batch_size * seq_len, embed_size, height, width)
        x = self.layers(x)
        x = x.view(batch_size, seq_len, self.hidden_dim[-1], self.convlstm_input_size[0], self.convlstm_input_size[1])
        _, pred = self.convlstm(x)
        final_state = pred[0][0].view(batch_size, self.latent_size)

        means = self.linear_means(final_state)
        logvar = self.linear_log_var(final_state)
        return means, logvar

class RevisedConvLSTMDecoder(BaseModule):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(RevisedConvLSTMDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvLSTMDecoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.

            layer_dim_list format: [input_dim, hidden_dim, kernel_size, num_layers, seq_len, convlstm_input_size]
        '''
        self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, self.seq_len, self.convlstm_input_size = layer_dim_list[0]
        self.kernel_size = tuple(self.kernel_size)

        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()
                                
        for i, dim_tuple in enumerate(layer_dim_list[1:]):
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
            self.layers.add_module(name="ConvTranspose2d{:d}".format(i+1), 
                                    module=nn.ConvTranspose2d(*dim_tuple))

        self.convlstm = ConvLSTM(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                kernel_size=self.kernel_size,
                                num_layers=self.num_layers, 
                                batch_first=True)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.repeat(1, self.seq_len)
        x = x.view(batch_size, self.seq_len, self.input_dim, self.convlstm_input_size[0], self.convlstm_input_size[1])
        x, _ = self.convlstm(x)

        x = x[0].view(batch_size * self.seq_len, self.hidden_dim[-1], self.convlstm_input_size[0], self.convlstm_input_size[1])
        x = self.layers(x)
        [_, embed_size, height, width] = x.size()
        x = x.view(batch_size, self.seq_len, embed_size, height, width)
        return x

class STGCNEncoder(BaseModule):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(STGCNEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMEncoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.

            layer_dim_list format: [num_nodes, num_features, num_input_time_steps]
        '''
        self.num_nodes, self.num_features, self.num_timesteps_input = layer_dim_list[0]

        num_layers = len(layer_dim_list) - 1

        block_l1 = layer_dim_list[1]
        block_l2 = layer_dim_list[2]
        block_l3 = layer_dim_list[3]
        self.block1 = STGCNBlock(in_channels=self.num_features, out_channels=block_l1[1],
                                 spatial_channels=block_l1[2], num_nodes=self.num_nodes)
        self.block2 = STGCNBlock(in_channels=block_l2[0], out_channels=block_l2[1],
                                 spatial_channels=block_l2[2], num_nodes=self.num_nodes)
        self.last_temporal = TimeBlock(in_channels=block_l3[0], out_channels=block_l3[1], 
                                 kernel_size=block_l3[2])

        
        self.latent_size = self.num_timesteps_input * self.num_nodes * block_l3[1]
        self.linear_means = nn.Linear(self.latent_size, self.latent_size)
        self.linear_log_var = nn.Linear(self.latent_size, self.latent_size)
        # self.weight_init()

    def forward(self, x):
        A_hat, x = x
        [batch_size, num_nodes, num_timesteps_input, num_features] = x.size()
        out1 = self.block1(x, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = out3.reshape((out3.shape[0], -1))

        means = self.linear_means(out4)
        logvar = self.linear_log_var(out4)
        return means, logvar

class STGCNDecoder(BaseModule):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        super(STGCNDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvLSTMDecoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.

            layer_dim_list format: [num_nodes, num_features, num_timesteps_output]
        '''
        self.num_nodes, self.num_features, self.num_timesteps_output = layer_dim_list[0]

        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        block_l1 = layer_dim_list[1]
        block_l2 = layer_dim_list[2]
        block_l3 = layer_dim_list[3]

        self.last_temporal = TimeBlock(in_channels=block_l1[0], out_channels=block_l1[1], 
                                        kernel_size=block_l1[2])
        self.block1 = STGCNBlock(in_channels=block_l2[0], out_channels=block_l2[1],
                                 spatial_channels=block_l2[2], num_nodes=self.num_nodes)
        self.block2 = STGCNBlock(in_channels=block_l3[0], out_channels=block_l3[1],
                                 spatial_channels=block_l3[2], num_nodes=self.num_nodes)

        
        self.latent_size = self.num_timesteps_output * self.num_nodes * block_l1[0]
        
    
    def forward(self, x):
        A_hat, x = x
        batch_size = x.size()[0]
        # x = x.repeat(1, self.num_timesteps_output)
        x = x.view(batch_size, self.num_nodes, self.num_timesteps_output, self.num_features)   
        x = self.last_temporal(x)
        out1 = self.block1(x, A_hat)
        out2 = self.block2(out1, A_hat)
        
        return out2


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias= True)

class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual # short cut

        return out

class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn = False):
        super(_bn_relu_conv, self).__init__()
        self.has_bn = bn
        #self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)

        return x

class STResNetEncoder(nn.Module):
    def __init__(self, layer_dim_list, activation=nn.ReLU, act_param=[]):
        '''
            C - Temporal Closeness
            P - Period
            T - Trend
            conf = (len_seq, nb_flow, map_height, map_width)
            external_dim
        '''

        super(STResNetEncoder, self).__init__()

        c_conf, p_conf, t_conf, (first_index, second_index), (external_dim, nb_residual_unit, latent_size) = layer_dim_list

        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf
        self.latent_size = latent_size
        self.first_index = first_index
        self.second_index = second_index

        self.nb_flow, self.map_height, self.map_width = t_conf[1], t_conf[2], t_conf[3]

        self.relu = torch.relu
        self.tanh = torch.tanh
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.qr_nums = len(self.quantiles)

        if self.c_conf is not None:
            self.c_way = self.make_one_way(in_channels = self.c_conf[0] * self.nb_flow)

        # Branch p
        if self.p_conf is not None:
            self.p_way = self.make_one_way(in_channels = self.p_conf[0] * self.nb_flow)

        # Branch t
        if self.t_conf is not None:
            self.t_way = self.make_one_way(in_channels = self.t_conf[0] * self.nb_flow)

        self.linear_means = nn.Linear(2 * self.map_height * self.map_width, self.latent_size)
        self.linear_log_var = nn.Linear(2 * self.map_height * self.map_width, self.latent_size)

    def make_one_way(self, in_channels):

        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels = in_channels, out_channels = 64)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter = 64, repetations = self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels = 64, out_channels = 2)),
            ('FusionLayer', TrainableEltwiseLayer(n = self.nb_flow, h = self.map_height, w = self.map_width))
        ]))

    def forward(self, x):
        input_c = x[:, :self.first_index]
        input_p = x[:, self.first_index:self.second_index]
        input_t = x[:, self.second_index:]
        # Three-way Convolution
        main_output = 0

        if self.c_conf is not None:
            input_c = input_c.view(-1, self.c_conf[0]*2, self.map_height, self.map_width)
            out_c = self.c_way(input_c)
            main_output += out_c
        if self.p_conf is not None:
            input_p = input_p.view(-1, self.p_conf[0]*2, self.map_height, self.map_width)
            out_p = self.p_way(input_p)
            main_output += out_p
        if self.t_conf is not None:
            input_t = input_t.view(-1, self.t_conf[0]*2, self.map_height, self.map_width)
            out_t = self.t_way(input_t)
            main_output += out_t

        main_output = main_output.view(-1, 2 * self.map_height * self.map_width)
        means = self.linear_means(main_output)
        logvar = self.linear_log_var(main_output)
        return means, logvar

class STResNetDecoder(nn.Module):
    def __init__(self, c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32),
                 t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
        '''
            C - Temporal Closeness
            P - Period
            T - Trend
            conf = (len_seq, nb_flow, map_height, map_width)
            external_dim
        '''

        super(STResNetDecoder, self).__init__()

        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf

        self.nb_flow, self.map_height, self.map_width = t_conf[1], t_conf[2], t_conf[3]

        self.relu = torch.relu
        self.tanh = torch.tanh
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.qr_nums = len(self.quantiles)

        if self.c_conf is not None:
            self.c_way = self.make_one_way(in_channels = self.c_conf[0] * self.nb_flow)

        # Branch p
        if self.p_conf is not None:
            self.p_way = self.make_one_way(in_channels = self.p_conf[0] * self.nb_flow)

        # Branch t
        if self.t_conf is not None:
            self.t_way = self.make_one_way(in_channels = self.t_conf[0] * self.nb_flow)

        # Operations of external component
        if self.external_dim != None and self.external_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.external_dim, 10, bias = True)),
                ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, self.nb_flow * self.map_height * self.map_width, bias = True)),
                ('relu2', nn.ReLU()),
            ]))

    def make_one_way(self, in_channels):

        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels = in_channels, out_channels = 64)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter = 64, repetations = self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels = 64, out_channels = 2)),
            ('FusionLayer', TrainableEltwiseLayer(n = self.nb_flow, h = self.map_height, w = self.map_width))
        ]))

    def forward(self, input_c, input_p, input_t, input_ext):
        # Three-way Convolution
        main_output = 0
        if self.c_conf is not None:
            input_c = input_c.view(-1, self.c_conf[0]*2, self.map_height, self.map_width)
            out_c = self.c_way(input_c)
            main_output += out_c
        if self.p_conf is not None:
            input_p = input_p.view(-1, self.p_conf[0]*2, self.map_height, self.map_width)
            out_p = self.p_way(input_p)
            main_output += out_p
        if self.t_conf is not None:
            input_t = input_t.view(-1, self.t_conf[0]*2, self.map_height, self.map_width)
            out_t = self.t_way(input_t)
            main_output += out_t

        # fusing with external component
        if self.external_dim != None and self.external_dim > 0:
            # external input
            external_output = self.external_ops(input_ext)
            external_output = self.relu(external_output)
            external_output = external_output.view(-1, self.nb_flow, self.map_height, self.map_width)
            #main_output = torch.add(main_output, external_output)
            main_output += external_output

        else:
            print('external_dim:', external_dim)
        main_output = self.tanh(main_output)

        return main_output

