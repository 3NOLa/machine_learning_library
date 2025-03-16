#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include "matrix.h"
#include "network.h"

void test_matrix_operations() {
    // Create matrices
    Matrix* A = matrix_zero_create(2, 2);

    // Set values
    matrix_set(A, 0, 0, 1.0); 

    Matrix* B = matrix_zero_create(2, 2);

    Matrix* C = matrix_add(A, B);

    printf("Addition Result:\n");
    for (int i = 0; i < C->rows; i++) {
        for (int j = 0; j < C->cols; j++) {
            printf("[%lf] ", matrix_get(C, i, j));
        }
        printf("\n");
    }

    Matrix* dot = matrix_mul(A, B);
    printf("Addition Result:\n");
    for (int i = 0; i < dot->rows; i++) {
        for (int j = 0; j < dot->cols; j++) {
            printf("[%lf] ", matrix_get(dot, i, j));
        }
        printf("\n");
    }

    // Free matrices
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);

}

void test_neural_net()
{
    Matrix* data = matrix_random_create(1, 200);
    int layerssize[5] = { 10,5,3,2,2 };
    ActivationType functions[5] = { RELu,Sigmoid,Tanh,RELu,Sigmoid };
    network* net = network_create(5, layerssize, 200, functions, 0.01);

    Matrix* output = forwardPropagation(net, data);
    for (int i = 0; i < output->rows; i++)
    {
        for (int j = 0; j < output->cols;j++)
        {
            printf("[%lf]", matrix_get(output, i, j));
        }
    }
}
int main() {
    //test_matrix_operations();
    test_neural_net();
    return 0;
}
