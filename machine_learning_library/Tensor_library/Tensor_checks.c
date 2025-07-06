#include "Tensor_HEADER.h"

int main(){
    srand(time(NULL));
    int shape[] = {2,2};
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

    Tensor* k = tensor_mmul(a, b, false);
    fprintf(stderr, "k: ");
    //tensor_print(k);
    fprintf(stderr, " \n");

    tensor_backward(e);
    fprintf(stderr, "a grad: ");
    for (int i = 0; i < a->count; i++){
        fprintf(stderr,"%f\t,", a->grad[i]);
    }

    fprintf(stderr, " \n");
    fprintf(stderr, "b grad: ");
    for (int i = 0; i < a->count; i++){
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
    for (int i = 0; i < a->count; i++){
        fprintf(stderr,"%f\t,", e->grad[i]);
    }
    fprintf(stderr, " \n");


    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(e);
    tensor_free(h);
    tensor_free(k); 
    tensor_free(eh);   
}