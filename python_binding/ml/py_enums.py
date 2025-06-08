from python_binding.cbinding.tasks import lib
from enum import IntEnum


class OptimizerType(IntEnum):
    SGD = lib.SGD
    SGDM = lib.SGDM
    NESTEROV = lib.NESTEROV
    RMSPROP = lib.RMSPROP
    ADAM = lib.ADAM


class InitializerType(IntEnum):
    RandomNormal = lib.RandomNormal
    RandomUniform = lib.RandomUniform
    XavierNormal = lib.XavierNormal
    XavierUniform = lib.XavierUniform
    HeNormal = lib.HeNormal
    HeUniform = lib.HeUniform
    LeCunNormal = lib.LeCunNormal
    LeCunUniform = lib.LeCunUniform
    Orthogonal = lib.Orthogonal
    Sparse = lib.Sparse


class ActivationType(IntEnum):
    RELU = lib.RELU
    LEAKY_RELU = lib.LEAKY_RELU
    SIGMOID = lib.SIGMOID
    TANH = lib.TANH
    LINEAR = lib.LINEAR
    GELU = lib.GELU
    SWISH = lib.SWISH


class LossType(IntEnum):
    MSE = lib.MSE
    MAE = lib.MAE
    Binary_Cross_Entropy = lib.Binary_Cross_Entropy
    Categorical_Cross_Entropy = lib.Categorical_Cross_Entropy
    Huber_Loss = lib.Huber_Loss


class LayerType(IntEnum):
    LAYER_DENSE = lib.LAYER_DENSE
    LAYER_RNN = lib.LAYER_RNN
    LAYER_LSTM = lib.LAYER_LSTM