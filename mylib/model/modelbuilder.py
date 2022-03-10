from .module.baseModel import *
from .module.utils import *

def get_encoder(mode, encoder_layer_dims, activation, act_param):
    if mode[0] == 'conv':
        temp_encoder = ConvEncoder(encoder_layer_dims, activation, act_param)
    elif mode[0] == 'linear':
        temp_encoder = BaseEncoder(encoder_layer_dims, activation, act_param)
    elif mode[0] == 'series':
        temp_encoder = LSTMEncoder(encoder_layer_dims, activation, act_param)
    elif mode[0] == 'spatio_temporal':
        conv_mode = mode[1]
        if conv_mode == 'original':
            temp_encoder = ConvLSTMEncoder(encoder_layer_dims, activation, act_param)
        elif conv_mode == 'revised':
            temp_encoder = RevisedConvLSTMEncoder(encoder_layer_dims, activation, act_param)
    elif mode[0] == 'stgcn':
        temp_encoder = STGCNEncoder(encoder_layer_dims, activation, act_param)
    elif mode[0] == 'st-resnet':
        temp_encoder = STResNetEncoder(encoder_layer_dims, activation, act_param)

    return temp_encoder

def get_decoder(mode, decoder_layer_dims, activation, act_param):
    if mode[0] == 'conv':
        temp_decoder = ConvDecoder(decoder_layer_dims, activation, act_param)
    elif mode[0] == 'linear':
        temp_decoder = BaseDecoder(decoder_layer_dims, activation, act_param)
    elif mode[0] == 'series':
        temp_decoder = LSTMDecoder(decoder_layer_dims, activation, act_param)
    elif mode[0] == 'spatio_temporal':
        conv_mode = mode[1]
        if conv_mode == 'original':
            temp_decoder = ConvLSTMDecoder(decoder_layer_dims, activation, act_param)
        elif conv_mode == 'revised':
            temp_decoder = RevisedConvLSTMDecoder(decoder_layer_dims, activation, act_param)
    elif mode[0] == 'stgcn':
        temp_decoder = STGCNDecoder(decoder_layer_dims, activation, act_param)
    elif mode[0] == 'st-resnet':
        temp_decoder = RevisedConvLSTMDecoder(decoder_layer_dims, activation, act_param)

    return temp_decoder