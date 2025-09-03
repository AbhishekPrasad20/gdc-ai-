#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;     // Raw pointer to flattened data
    int* shape;      // Array describing tensor dimensions
    int dims;        // Number of dimensions (e.g. 2 for matrix)
    int size;        // Total number of elements (product of shape[])
} Tensor;

// Create a new tensor given shape (e.g., {2, 3}, dims = 2)
Tensor* tensor_create(int* shape, int dims);

void tensor_print(Tensor* t);
void tensor_add(Tensor* t1, Tensor* t2, Tensor* result);
void tensor_subtract(Tensor* t1, Tensor* t2, Tensor* result);
void tensor_multiply(Tensor* t1, Tensor* t2, Tensor* result);
void tensor_free(Tensor* t);
void tensor_dot(Tensor* t1, Tensor* t2, Tensor* result);
// Set or get flat index values
void tensor_set(Tensor* t, int index, float value);
void tensor_relu(Tensor* t);
void tensor_dense_forward(Tensor* input, Tensor* weights, Tensor* bias, Tensor* output);
void compute_loss(Tensor* logits, Tensor* labels, Tensor* loss);
void tensor_softmax(Tensor* t);    
void loss_gradient(Tensor* logits, Tensor* labels, Tensor* gradient);
void tensor_subtract_scalar(Tensor* t, float scalar, Tensor* result);
void tensor_argmax(Tensor* t, int* index);
void index_to_char(int index, int* shape, int dims, char* output);
float tensor_get(Tensor* t, int index);
void tensor_subtract_scaled(Tensor* t, Tensor* grad, float lr);
void cross_entropy_loss(Tensor* logits, Tensor* labels, Tensor* loss);
void tensor_conv2d(Tensor* input, Tensor* kernel, Tensor* output);
// Debug print (optional, but super useful)
void tensor_print(Tensor* t);

#endif
