#include "network.h"

network* network_create(int layerAmount, int* layersSize, int input_dim, ActivationType* activations, double learnningRate)
{
	network* net = (network*)malloc(sizeof(network));
	net->learnningRate = learnningRate;
	net->layerAmount = layerAmount;

	net->layers = (layer**)malloc(sizeof(layer*) * layerAmount);
	net->layersSize = (int*)malloc(sizeof(int) * layerAmount);
	int lastLayerSize = input_dim;
	for (int i = 0; i < layerAmount; i++)
	{
		net->layers[i] = layer_create(layersSize[i], lastLayerSize, activations[i]);
		net->layersSize[i] = lastLayerSize = layersSize[i];
	}

	return net; 
}

network* network_create_empty()
{
    network* net = (network*)malloc(sizeof(network));
    net->learnningRate = 0.0;
    net->layerAmount = 0;
    net->layers = NULL;
    net->layersSize = NULL;

    return net;
}

void add_layer(network* net, int layerSize,ActivationType Activationfunc, int input_dim)
{
    input_dim = (input_dim > 0) ? input_dim : net->layersSize[net->layerAmount];
    net->layersSize = (int*)realloc(net->layersSize, sizeof(int)* (net->layerAmount+1));
    net->layersSize[net->layerAmount+1] = layerSize;
    net->layers = (layer**)realloc(net->layers, sizeof(layer*) * (net->layerAmount + 1));
    net->layers[net->layerAmount++] = layer_create(layerSize, input_dim, Activationfunc);
}

void network_free(network* net)
{
    if (net) {
        for (int i = 0; i < net->layersSize; i++)
            layer_free(net->layers[i]);
        free(net->layers);
        free(net);
    }

}

double forwardPropagation(network* net,Matrix* data)
{
    double y_hat = 0;
    Matrix* current_output,*current_input=data;
    
    for (int i = 0; i < net->layerAmount; i++)
    {
        current_output = layer_forward(net->layers[i], current_input);
        
        if (i > 0)
            matrix_free(current_input);
        current_input = current_output;
    }
}

