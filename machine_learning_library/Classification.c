#include "classification.h"

ClassificationNetwork* ClassificationNetwork_create(int layerAmount, int* layersSize, int* input_shape, int input_dim, ActivationType* activations, double learnningRate, LossType lossFunction, char** class_names, int num_classes, Tensor* classes)
{
    if (layerAmount <= 0 || !layersSize || !activations || input_dim <= 0 || !classes || !class_names || num_classes <= 0) {
        fprintf(stderr, "Error: Invalid parameters in ClassificationNetwork_create\n");
        return NULL;
    }

    ClassificationNetwork* Cnet = (ClassificationNetwork*)malloc(sizeof(ClassificationNetwork));
    if (!Cnet) {
        fprintf(stderr, "Error: Memory allocation failed for network in ClassificationNetwork_create_net\n");
        return NULL;
    }

    Cnet->num_classes = num_classes;
    Cnet->one_hot_encode = one_hot_encode(num_classes); 
    Cnet->net = network_create(layerAmount,layersSize,input_dim, input_shape,activations,learnningRate, lossFunction); //cant free the network
    if(!Cnet->net){
        fprintf(stderr, "Error: Memory allocation failed for network in ClassificationNetwork_create_net\n");
        return NULL;
    }

    Cnet->class_names = (char**)malloc(sizeof(char*) * Cnet->num_classes);
    if (!Cnet->class_names) {
        fprintf(stderr, "Error: Memory allocation failed for class names\n");
        return;
    }

    for (int i = 0; i < Cnet->num_classes; i++) {
        if (class_names[i]) {
            Cnet->class_names[i] = _strdup(class_names[i]);
            if (!Cnet->class_names[i]) {
                fprintf(stderr, "Error: Memory allocation failed for class name %d\n", i);
            }
        }
        else {
            Cnet->class_names[i] = NULL;
        }
    }

    return Cnet;
}

ClassificationNetwork* ClassificationNetwork_create_net(network* net, char** class_names, int num_classes, Tensor* classes)
{
    if (!classes || !net || !class_names || num_classes <= 0) {
        fprintf(stderr, "Error: Invalid parameters in ClassificationNetwork_create_net\n");
        return NULL;
    }

    ClassificationNetwork* Cnet = (ClassificationNetwork*)malloc(sizeof(ClassificationNetwork));
    if (!Cnet) {
        fprintf(stderr, "Error: Memory allocation failed for network in ClassificationNetwork_create_net\n");
        return NULL;
    }

    Cnet->num_classes = num_classes;
    Cnet->net = net; //cant free the network
    Cnet->one_hot_encode = one_hot_encode(num_classes);

    Cnet->class_names = (char**)malloc(sizeof(char*) * Cnet->num_classes);
    if (!Cnet->class_names) {
        fprintf(stderr, "Error: Memory allocation failed for class names\n");
        return;
    }

    for (int i = 0; i < Cnet->num_classes; i++) {
        if (class_names[i]) {
            Cnet->class_names[i] = _strdup(class_names[i]);
            if (!Cnet->class_names[i]) {
                fprintf(stderr, "Error: Memory allocation failed for class name %d\n", i);
            }
        }
        else {
            Cnet->class_names[i] = NULL;
        }
    }

    return Cnet;
}

Tensor* one_hot_encode(int num_classes)
{
    if(num_classes < 2) {
        fprintf(stderr, "Error: num_classes too small failed in one_hot_encode\n");
        return NULL;
    }

    Tensor* one_hot_encode = tensor_identity_create(num_classes);

    return one_hot_encode;
}


void classification_info_free(ClassificationNetwork* info)
{
    if (!info) {
        fprintf(stderr, "Error: Cant free a null in classification_info_free\n");
        return NULL;
    }
    if(info->net)neuron_free(info->net);
    if(info->one_hot_encode)tensor_free(info->one_hot_encode);
    
    if (info->class_names) {
        for (int i = 0; i < info->num_classes; i++) {
            if (info->class_names[i]) {
                free(info->class_names[i]);
            }
        }
        free(info->class_names);
    }
    free(info);
}

int get_predicted_class(Tensor* network_output)
{
    if (!network_output) {
        fprintf(stderr, "Error: NULL network output in get_predicted_class\n");
        return -1;
    }

    // Find the index of the maximum value in the output
    double max_val = network_output->data[0];
    int max_idx = 0;

    for (int i = 1; i < network_output->count; i++) {
        if (network_output->data[i] > max_val) {
            max_val = network_output->data[i];
            max_idx = i;
        }
    }

    return max_idx;
}