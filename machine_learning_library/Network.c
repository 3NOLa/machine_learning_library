#include "classification.h"
#include "optimizers.h"
#include "cfgparse.h"

network* network_create(int layerAmount, int* layersSize, int input_dims, int* input_shape, ActivationType* activations, float  learnningRate, LossType lossFunction,LayerType type)
{
    if (layerAmount <= 0 || !layersSize || !activations || input_dims <= 0 || !input_shape) {
        fprintf(stderr, "Error: Invalid parameters in network_create\n");
        return NULL;
    }

    network* net = (network*)malloc(sizeof(network));
    if (!net) {
        fprintf(stderr, "Error: Memory allocation failed for network\n");
        return NULL;
    }

    net->learnningRate = learnningRate; 
    net->layerAmount = layerAmount;
    net->lossFunction = lossFunction;
    net->LossFuntionPointer = LossTypeMap(lossFunction);
    net->LossDerivativePointer = LossTypeDerivativeMap(lossFunction);
    net->input_dims = input_dims;
    net->ltype = type;
    net->otype = SGD;


    net->input_shape = (int*)malloc(sizeof(int) * input_dims);
    if (!net->input_shape) {
        fprintf(stderr, "Error: Memory allocation failed for input shape\n");
        free(net);
        return NULL;
    }

    for (int i = 0; i < input_dims; i++)
        net->input_shape[i] = input_shape[i];
    

    int total_input_size = 1;
    for (int i = 0; i < input_dims; i++) {
        total_input_size *= input_shape[i];
    }

    net->layers = (layer**)malloc(sizeof(layer*) * layerAmount);
    if (!net->layers) {
        fprintf(stderr, "Error: Memory allocation failed for layers array\n");
        free(net);
        return NULL;
    }

    net->layersSize = (int*)malloc(sizeof(int) * layerAmount);
    if (!net->layersSize) {
        fprintf(stderr, "Error: Memory allocation failed for layersSize array\n");
        free(net->layers);
        free(net);
        return NULL;
    }

    int lastLayerSize = total_input_size;
    for (int i = 0; i < layerAmount; i++) {
        net->layers[i] = general_layer_Initialize(type,layersSize[i], lastLayerSize, activations[i]);
        if (!net->layers[i]) {
            fprintf(stderr, "Error: Failed to create dense_layer %d\n", i);

            for (int j = 0; j < i; j++) {
                general_layer_free(net->layers[j]);
            }

            free(net->layersSize);
            free(net->layers);
            free(net->input_shape);
            free(net);
            return NULL;
        }

        net->layersSize[i] = layersSize[i];
        lastLayerSize = layersSize[i];
    }

    network_train_type(net);

    return net;
}

network* network_create_empty()
{
    network* net = (network*)malloc(sizeof(network));
    if (!net) {
        fprintf(stderr, "Error: Memory allocation failed for empty network\n");
        return NULL;
    }

    net->learnningRate = 0.0;
    net->layerAmount = 0;
    net->input_dims = 0;
    net->input_shape = NULL;
    net->layers = NULL;
    net->layersSize = NULL;
    net->LossDerivativePointer = NULL;
    net->lossFunction = MSE;
    net->train = NULL;
    net->ltype = LAYER_DENSE;

    return net;
}

int add_layer(network* net, int layerSize, ActivationType Activationfunc, int input_dim)
{
    if (!net) {
        fprintf(stderr, "Error: NULL network in add_layer\n");
        return 0;
    }

    if (layerSize <= 0) {
        fprintf(stderr, "Error: Invalid dense_layer size %d in add_layer\n", layerSize);
        return 0;
    }

    // Determine input dimension for the new layer
    int actual_input_dim;
    if (input_dim > 0) {
        actual_input_dim = input_dim;
    }
    else if (net->layerAmount > 0) {
        actual_input_dim = net->layersSize[net->layerAmount - 1];
    }
    else if (net->input_dims > 0) {
        // For the first layer, use the total size of the input tensor
        actual_input_dim = 1;
        for (int i = 0; i < net->input_dims; i++) {
            actual_input_dim *= net->input_shape[i];
        }
    }
    else {
        fprintf(stderr, "Error: Cannot determine input dimension for first dense_layer\n");
        return 0;
    }

    // Resize the layersSize array
    int* new_layersSize = (int*)realloc(net->layersSize, sizeof(int) * (net->layerAmount + 1));
    if (!new_layersSize) {
        fprintf(stderr, "Error: Memory reallocation failed for layersSize in add_layer\n");
        return 0;
    }
    net->layersSize = new_layersSize;

    // Resize the layers array
    layer** new_layers = (layer**)realloc(net->layers, sizeof(layer*) * (net->layerAmount + 1));
    if (!new_layers) {
        fprintf(stderr, "Error: Memory reallocation failed for layers in add_layer\n");
        return 0;
    }
    net->layers = new_layers;

    // Create the new layer
    net->layers[net->layerAmount] = general_layer_Initialize(net->ltype,layerSize, actual_input_dim, Activationfunc);
    if (!net->layers[net->layerAmount]) {
        fprintf(stderr, "Error: Failed to create dense_layer in add_layer\n");
        return 0;
    }

    // Update layer size and count
    net->layersSize[net->layerAmount] = layerSize;
    net->layerAmount++;

    return 1;
}

int add_created_layer(network* net, layer* l)
{
    if (!net || !l) {
        fprintf(stderr, "Error: NULL network or NULL layer in add_created_layer\n");
        return 0;
    }

    int* new_layersSize = (int*)realloc(net->layersSize, sizeof(int) * (net->layerAmount + 1));
    if (!new_layersSize) {
        fprintf(stderr, "Error: Memory reallocation failed for layersSize in add_layer\n");
        return 0;
    }
    net->layersSize = new_layersSize;

    // Resize the layers array
    layer** new_layers = (layer**)realloc(net->layers, sizeof(layer*) * (net->layerAmount + 1));
    if (!new_layers) {
        fprintf(stderr, "Error: Memory reallocation failed for layers in add_layer\n");
        return 0;
    }
    net->layers = new_layers;

    net->layers[net->layerAmount] = l;
    net->layersSize[net->layerAmount] = l->neuronAmount;
    net->layerAmount++;

    return 1;
}

int set_loss_function(network* net, LossType lossFunction)
{
    if (!net) {
        fprintf(stderr, "Error: NULL network in set_loss_function\n");
        return 0;
    }

    net->lossFunction = lossFunction;
    net->LossFuntionPointer = LossTypeMap(lossFunction);
    net->LossDerivativePointer = LossTypeDerivativeMap(lossFunction);
}

void set_network_optimizer(network* net, OptimizerType type)
{
    if (!net) {
        fprintf(stderr, "Error: NULL network in set_network_optimizer\n");
        return;
    }

    net->otype = type;
    for (int i = 0; i < net->layerAmount; i++)
        set_layer_optimizer(net->layers[i], type);
}

void network_free(network* net)
{
    if (net) {
        if (net->layers) {
            for (int i = 0; i < net->layerAmount; i++) {
                if (net->layers[i]) {
                    general_layer_free(net->layers[i]);
                }
            }
            free(net->layers);
        }

        if (net->layersSize) 
            free(net->layersSize);

        if (net->input_shape) 
            free(net->input_shape);

        free(net);
    }
}

void network_train_type(network* net)
{
    switch (net->ltype)
    {
    case LAYER_DENSE:
        net->train = train;
    case LAYER_RNN:
        net->train = rnn_train;
    case LAYER_LSTM:
        net->train = rnn_train;
    default:
        break;
    }
}

Tensor* forwardPropagation(network* net, Tensor* data)
{
    if (!net || !data) {
        fprintf(stderr, "Error: NULL network or data in forwardPropagation\n");
        return NULL;
    }

    if (net->layerAmount <= 0) {
        fprintf(stderr, "Error: Network has no layers in forwardPropagation\n");
        return NULL;
    }

    Tensor* current_output = NULL;
    Tensor* current_input = data;
    
    // If input is not 1D, we need to flatten it first
    Tensor* flattened_input = NULL;
    if (data->dims > 1) {
        flattened_input = tensor_flatten(data);
        if (!flattened_input) {
            fprintf(stderr, "Error: Failed to flatten input tensor\n");
            return NULL;
        }
        current_input = flattened_input;
    }

    for (int i = 0; i < net->layerAmount; i++) {
        current_output = net->layers[i]->forward(net->layers[i], current_input);
        if (!current_output) {
            fprintf(stderr, "Error: Layer %d forward propagation failed\n", i);

            // Free previous intermediate outputs
            if (i > 0 && current_input != data) {
                tensor_free(current_input);
            }

            return NULL;
        }

        // Free previous intermediate output (but not the original input)
        if (i > 0 && current_input != data) 
            tensor_free(current_input);

        current_input = current_output;
    }

    if (flattened_input) 
        tensor_free(flattened_input);

    return current_output;
}

int backpropagation(network* net, Tensor* predictions, Tensor* targets)
{
    if (!net || !predictions || !targets) {
        fprintf(stderr, "Error: NULL parameters in backpropagation\n");
        return 0;
    }

    if (net->layerAmount <= 0) {
        fprintf(stderr, "Error: Network has no layers in backpropagation\n");
        return 0;
    }

    // Calculate error derivatives
    Tensor* output_gradients = net->LossDerivativePointer(net, targets);
    if (!output_gradients) {
        fprintf(stderr, "Error: Failed to compute error derivatives in backpropagation\n");
        return 0;
    }

    // Gradients to be passed to each layer
    Tensor* current_gradients = output_gradients;
    Tensor* new_gradients = NULL;

    // Backpropagate through each layer in reverse order
    for (int i = net->layerAmount - 1; i >= 0; i--) {
        new_gradients = net->layers[i]->backward(net->layers[i], current_gradients);
        if (!new_gradients) {
            fprintf(stderr, "Error: Layer %d backpropagation failed\n", i);

            // Free current gradients if they're not the output gradients (which we'll free later)
            if (current_gradients != output_gradients) {
                tensor_free(current_gradients);
            }

            tensor_free(output_gradients);
            return 0;
        }

        // Free previous gradients (except the original output gradients which we'll free after the loop)
        if (current_gradients != output_gradients) {
            tensor_free(current_gradients);
        }

        current_gradients = new_gradients;
    }

    // Free the final gradients and output gradients
    tensor_free(current_gradients);
    tensor_free(output_gradients);

    return 1;
}

void network_update(network* net) {
    for (int i = 0; i < net->layerAmount; i++) {
        net->layers[i]->update(net->layers[i], net->learnningRate);
    }
}

void network_zero_grad(network* net)
{
    for (int i = 0; i < net->layerAmount; i++) {
        net->layers[i]->zero_grad(net->layers[i]);
        }
}

void network_opt_init(network* net, Initializer* init, initializerType type)
{
    for (int i = 0; i < net->layerAmount; i++) {
        net->layers[i]->opt_init(net->layers[i],init,type);
    }
}

void network_reset_state(network* net) {
    for (int i = 0; i < net->layerAmount; i++) {
        if (net->layers[i]->reset_state)
            net->layers[i]->reset_state(net->layers[i]);
    }
}

int save_model(const network* net, const char* cfg_path, const char* weights_path)
{
    FILE* wfp = NULL;
    FILE* cfp = NULL;

    errno_t werr = fopen_s(&wfp, weights_path, "wb");
    errno_t cerr = fopen_s(&cfp, cfg_path, "w");

    if (werr != 0) {
        fprintf(stderr, "Error opening weights file: %s\n", weights_path);
    }
    if (cerr != 0) {
        fprintf(stderr, "Error opening config file: %s\n", cfg_path);
    }

    if (werr != 0 || cerr != 0) {
        if (wfp) fclose(wfp);  // clean up any open file
        return 1;
    }

    fprintf(cfp, "# Network model file\n");
    fprintf(cfp, "\n# Network\n");
    fprintf(cfp, "layers amount = %d\n", net->layerAmount);
    fprintf(cfp, "learning rate = %f\n", net->learnningRate);
    fprintf(cfp, "Optimizer Type = %d\n", net->otype);
    fprintf(cfp, "Network Type = %d\n", net->ltype);
    fprintf(cfp, "Loss Type = %d\n", net->lossFunction);


    for (int i = 0; i < net->layerAmount; i++)
    {
        fprintf(cfp, "\n# Layer %d\n", i);
        save_layer_model(wfp, cfp, net->layers[i]);
    }

    if (wfp) fclose(wfp);
    if (cfp) fclose(cfp);
    fprintf(stderr, "saved network model success");
}

void load_weights_model(network* net, FILE* wfp) {
    for (int i = 0; i < net->layerAmount; i++)
    {
        load_layer_weights_model(wfp, net->layers[i]);
    }
}

network* load_model(const char* cfg_path, const char* weights_path) {
    FILE* wfp = NULL;
    FILE* cfp = NULL;

    errno_t werr = fopen_s(&wfp, weights_path, "rb");
    errno_t cerr = fopen_s(&cfp, cfg_path, "r");

    if (werr != 0) {
        fprintf(stderr, "Error opening weights file: %s\n", weights_path);
    }
    if (cerr != 0) {
        fprintf(stderr, "Error opening config file: %s\n", cfg_path);
    }

    ConfigMap* map = ConfigMapcreate(20);

    char line[MAX_LINE];
    while (fgets(line, sizeof(line), cfp)) {
        // Skip comments and empty lines
        char* trimmed = trim(line);
        if (trimmed[0] == '#' || trimmed[0] == '\0') continue;

        parse_cfg_line(trimmed, map);
    }

    if (cfp) fclose(cfp);
    // Fetch config values only once per field
    ConfigValues* layersVal = Configmap_get(map, "layers amount");
    ConfigValues* neuronsVal = Configmap_get(map, "neurons amount");
    ConfigValues* inputDimVal = Configmap_get(map, "Layer input dim");
    ConfigValues* shapeVal = Configmap_get(map, "Layer shape");
    ConfigValues* activationsVal = Configmap_get(map, "Activation type");
    ConfigValues* lrVal = Configmap_get(map, "learning rate");
    ConfigValues* lossVal = Configmap_get(map, "Loss Type");
    ConfigValues* typeVal = Configmap_get(map, "Network Type");

    // Convert to typed pointers
    int layersAmount = layersVal[0].i;
    int* neurons = config_values_to_int_array(neuronsVal, layersAmount);
    int inputDim = inputDimVal[0].i;
    int* inputShape = config_values_to_int_array(shapeVal,inputDim);
    ActivationType* activations = (ActivationType*)config_values_to_int_array(activationsVal, layersAmount);
    float learningRate = lrVal[0].f;
    LossType lossFunction = (LossType)lossVal[0].i;
    LayerType networkType = (LayerType)typeVal[0].i;

    // Print debug info
    fprintf(stderr, "Creating network with:\n");
    fprintf(stderr, "  layers amount       = %d\n", layersAmount);

    fprintf(stderr, "  neurons amount      = ");
    for (int i = 0; i < layersAmount; i++) {
        fprintf(stderr, "%d ", neurons[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "  input dim           = %d\n", inputDim);

    fprintf(stderr, "  input shape         = ");
    for (int i = 0; i < inputDim; i++) {
        fprintf(stderr, "%d ", inputShape[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "  activation types    = ");
    for (int i = 0; i < layersAmount; i++) {
        fprintf(stderr, "%d ", activations[i]);  // Replace with string if desired
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "  learning rate       = %f\n", learningRate);
    fprintf(stderr, "  loss type           = %d\n", lossFunction);
    fprintf(stderr, "  network type        = %d\n", networkType);

    // Create the network
    network* net = network_create(
        layersAmount,
        neurons,
        inputDim,
        inputShape,
        activations,
        learningRate,
        lossFunction,
        networkType
    );


    set_network_optimizer(net, (OptimizerType)Configmap_get(map, "Optimizer Type")[0].i);

    load_weights_model(net, wfp);

    return net;
}

float  train(network* net, Tensor* input, Tensor* target)
{
    if (!net || !input || !target) {
        fprintf(stderr, "Error: NULL parameters in train\n");
        return -1.0;  // Return negative error to indicate failure
    }

    network_zero_grad(net);

    // Forward pass
    Tensor* predictions = forwardPropagation(net, input);
    if (!predictions) {
        fprintf(stderr, "Error: Forward propagation failed in train\n");
        return -1.0;
    }

    // Calculate error
    float  error = net->LossFuntionPointer(net, target);

    // Backward pass
    if (!backpropagation(net, predictions, target)) {
        fprintf(stderr, "Error: Backpropagation failed in train\n");
        tensor_free(predictions);
        return -1.0;
    }

    // Free resources
    tensor_free(predictions);

    network_update(net);

    return error;
}

float  rnn_train(network* net, Tensor* input, Tensor* target, int timestamps)
{
    if (!net || !input || !target) {
        fprintf(stderr, "Error: NULL parameters in train\n");
        return -1.0;  // Return negative error to indicate failure
    }

    network_zero_grad(net);

    network_reset_state(net);

    Tensor* predictions = NULL;
    for (int i = 0; i < timestamps-1; i++)
    {
        Tensor* input_t = tensor_slice_range(input, i,i+1);
        input_t = tensor_flatten(input_t);

        Tensor* pred = forwardPropagation(net, input_t);
        tensor_free(input_t);

        if (predictions) tensor_free(predictions);  // free previous
        predictions = pred;
    }
    // Calculate error
    float  error = net->LossFuntionPointer(net, target);

    // Backward pass
    if (!backpropagation(net, predictions, target)) {
        fprintf(stderr, "Error: Backpropagation failed in train\n");
        tensor_free(predictions);
        return -1.0;
    }

    // Free resources
    tensor_free(predictions);
    network_update(net);

    return error;
}

void network_training(network* net, Tensor* input, Tensor* target, int epcho, int batch_size)
{
    for (int i = 0; i < epcho; i+= batch_size) {
        int current_batch = (batch_size > epcho - i) ? batch_size : epcho - i;

        Tensor* input_batch = tensor_slice_range(input, i, current_batch);
        Tensor* output_batch = tensor_slice_range(target, i, current_batch);

        float  error = train(net, input_batch, output_batch);

        if (i % (epcho/10) == 0) {
            printf("Iteration %d, Error: %.4f\n", i, error);
        }
    }
}