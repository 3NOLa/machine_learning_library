#pragma once
#include "network.h"
#include <string.h>

typedef struct {
    int num_classes;
    Tensor* one_hot_encode;
    char** class_names;
    network* net;
} ClassificationNetwork;

ClassificationNetwork* ClassificationNetwork_create(int layerAmount, int* layersSize, int* input_shape, int input_dim, ActivationType* activations, double learnningRate, LossType lossFunction, char** class_names, int num_classes, Tensor* classes);
ClassificationNetwork* ClassificationNetwork_create_net(network* net, char** class_names, int num_classes, Tensor* classes);
Tensor* one_hot_encode(int num_classes);
int get_predicted_class(Tensor* network_output);

void classification_info_free(ClassificationNetwork* info);
