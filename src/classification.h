#pragma once
#include "network.h"
#include "export.h"
#include <string.h>

EXPORT typedef struct {
    int num_classes;
    Tensor* one_hot_encode;
    char** class_names;
    network* net;
} ClassificationNetwork;

EXPORT ClassificationNetwork* ClassificationNetwork_create(int layerAmount, int* layersSize, int* input_shape, int input_dim, ActivationType* activations, float  learnningRate, LossType lossFunction, char** class_names, int num_classes, Tensor* classes, LayerType type);
EXPORT ClassificationNetwork* ClassificationNetwork_create_net(network* net, char** class_names, int num_classes, Tensor* classes);
EXPORT Tensor* one_hot_encode(int num_classes);
EXPORT void classification_info_free(ClassificationNetwork* info);

EXPORT int get_predicted_class(Tensor* network_output);
EXPORT void classification_network_training(ClassificationNetwork* Cnet);