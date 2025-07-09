#include "Tensor_HEADER.h"

int main(){
    srand(time(NULL));
    int shape[] = {4,4};
    Tensor *a = tensor_random_create(2,shape);
    fprintf(stderr, "\na: ");
    tensor_print(a);

    Tensor *b = tensor_random_create(2,shape);
    fprintf(stderr, "\nb: ");
    tensor_print(b);

    Tensor *c = tensor_add(a, b);
    fprintf(stderr, "\nc: ");
    tensor_print(c);

    Tensor *d = tensor_subtract(a, b);
    fprintf(stderr, "\nd: ");
    tensor_print(d);
    
    Tensor *e = tensor_multiply(a, b);
    fprintf(stderr, "\ne: ");
    tensor_print(e);
    
    Tensor *h = tensor_div(a, b);
    fprintf(stderr, "\nh: ");
    tensor_print(h);

    tensor_backward(e);
    fprintf(stderr, "\na grad: ");
    tensor_print_grad(a);

    fprintf(stderr, "\nb grad: ");
    tensor_print_grad(b);

    Tensor *eh = tensor_div(e, a);
    fprintf(stderr, "\neh: ");
    tensor_print(eh);
    tensor_backward(eh);

    fprintf(stderr, "\na again grad: ");
    tensor_print_grad(a);

    fprintf(stderr, "\ne grad: ");
    tensor_print_grad(e);


    Tensor *mat1 = tensor_random_create(3,(int []) {2,2,2});
    fprintf(stderr, "\nmat1: ");
    tensor_print(mat1);

    Tensor *mat2 = tensor_random_create(2,(int []) {2,4});
    fprintf(stderr, "\nmat2: ");
    tensor_print(mat2);

    Tensor* k = tensor_mmul(mat1, mat2, false);
    fprintf(stderr, "\nk: ");
    tensor_print(k);
    tensor_backward(k);

    fprintf(stderr, "\n mat1 grad for matmul: \n");
    tensor_print_grad(mat1);

    fprintf(stderr, "\n mat2 grad for matmul: \n");
    tensor_print_grad(mat2);

    Tensor *mat = tensor_random_create(3,(int []) {2,2,5});
    fprintf(stderr, "\nmat: ");
    tensor_print(mat);

    Tensor *mat_t = tensor_transpose(mat);
    fprintf(stderr, "\n mat_t: \n");
    tensor_print(mat_t);

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
    tensor_free(mat_t); 
}