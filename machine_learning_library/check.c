/* #define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include "matrix.h"
#include "network.h"

void test_matrix_operations() {
    // Create matrices
    Tensor* A = tensor_zero_create(2, 2);

    // Set values
    tensor_set(A, 0, 0, 1.0); 

    Tensor* B = tensor_zero_create(2, 2);

    Tensor* C = matrix_add(A, B);

    printf("Addition Result:\n");
    for (int i = 0; i < C->rows; i++) {
        for (int j = 0; j < C->cols; j++) {
            printf("[%lf] ", tensor_get_index(C, i, j));
        }
        printf("\n");
    }

    Tensor* dot = matrix_mul(A, B);
    printf("Addition Result:\n");
    for (int i = 0; i < dot->rows; i++) {
        for (int j = 0; j < dot->cols; j++) {
            printf("[%lf] ", tensor_get_index(dot, i, j));
        }
        printf("\n");
    }

    // Free matrices
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);

}

void test_neural_net()
{
    Tensor* data = tensor_random_create(1, 200);
    int layerssize[5] = { 10,5,3,2,2 };
    ActivationType functions[5] = { RELu,Sigmoid,Tanh,RELu,Sigmoid };
    network* net = network_create(5, layerssize, 200, functions, 0.01);

    Tensor* output = forwardPropagation(net, data);
    for (int i = 0; i < output->rows; i++)
    {
        for (int j = 0; j < output->cols;j++)
        {
            printf("[%lf]", tensor_get_index(output, i, j));
        }
    }
}

void test_trainnig_neural_net()
{
    Tensor* data = tensor_random_create(10, 100);
    Tensor* y_real = tensor_random_create(1, 1);
    y_real->data[0] = 1;
    int layerssize[5] = { 10,5,3,2,1};
    ActivationType functions[5] = { RELu,Sigmoid,Tanh,RELu,Sigmoid };
    network* net = network_create(5, layerssize, 100, functions, 0.1);
    
    for (int i = 0; i < 30; i++)
    {
        double error = train(net, data, y_real);
        printf("loop %d error [%lf]\n", i, error);
    }

}
//int main() {
    //test_matrix_operations();
    //test_neural_net();
//    test_trainnig_neural_net();
//    return 0;
//}*/
