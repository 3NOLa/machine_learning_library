#include "Tensor_HEADER.h"

int main(){
    srand(time(NULL));
    int shape[] = {8,8};
    Tensor *a = tensor_random_create(2,shape);
    fprintf(stderr, "a: ");
    tensor_print(a);
    fprintf(stderr, " \n");

    Tensor *b = tensor_random_create(2,shape);
    fprintf(stderr, ": ");
    tensor_print(b);
    fprintf(stderr, " \n");

    Tensor *c = tensor_add(a, b);
    fprintf(stderr, "c: ");
    tensor_print(c);
    fprintf(stderr, " \n");

    Tensor *d = tensor_subtract(a, b);
    fprintf(stderr, "d: ");
    tensor_print(d);
    fprintf(stderr, " \n");
    
    Tensor *e = tensor_multiply(a, b);
    fprintf(stderr, "e: ");
    tensor_print(e);
    fprintf(stderr, " \n");
    
    Tensor *h = tensor_div(a, b);
    fprintf(stderr, "h: ");
    tensor_print(h);
    fprintf(stderr, " \n");

    tensor_backward(e);
    fprintf(stderr, "a grad: ");
    for (int i = 0; i < a->count; i++){
        fprintf(stderr,"%f\t,", a->grad[i]);
    }

    fprintf(stderr, " \n");
    fprintf(stderr, "b grad: ");
    for (int i = 0; i < b->count; i++){
        fprintf(stderr,"%f\t,", b->grad[i]);
    }

    fprintf(stderr, " \n");
    Tensor *eh = tensor_div(e, a);
    fprintf(stderr, "eh: ");
    tensor_print(eh);
    tensor_backward(eh);

    fprintf(stderr, "a again grad: ");
    for (int i = 0; i < a->count; i++){
        fprintf(stderr,"%f\t,", a->grad[i]);
    }

    fprintf(stderr, " \n");
    fprintf(stderr, "e grad: ");
    for (int i = 0; i < e->count; i++){
        fprintf(stderr,"%f\t,", e->grad[i]);
    }


    Tensor *mat1 = tensor_random_create(2,(int []) {3,5});
    fprintf(stderr, "mat1: ");
    tensor_print(mat1);
    fprintf(stderr, " \n");

    Tensor *mat2 = tensor_random_create(2,(int []) {5,4});
    fprintf(stderr, "mat2: ");
    tensor_print(mat2);
    fprintf(stderr, " \n");

    Tensor* k = tensor_mmul(mat1, mat2, false);
    fprintf(stderr, "k: ");
    tensor_print(k);
    tensor_backward(k);
    fprintf(stderr, " \n");

    fprintf(stderr, "\n mat1 grad for matmul: \n");
    for (int i = 0; i < mat1->count; i++){
        fprintf(stderr,"%f\t,", mat1->grad[i]);
    }


    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(e);
    tensor_free(h);
    tensor_free(k); 
    tensor_free(eh);  
    tensor_free(mat1); 
    tensor_free(mat2);   
}