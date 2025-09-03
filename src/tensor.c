#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"

Tensor* tensor_create(int* shape, int dims) {
    Tensor* t = (Tensor*) malloc(sizeof(Tensor));
    t->shape = (int*) malloc(sizeof(int) * dims);
    t->dims = dims;
    t->size = 1;

    for (int i = 0; i < dims; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    t->data = (float*) calloc(t->size, sizeof(float));
    return t;
}
void tensor_print(Tensor* t) {
    printf("Tensor shape: [");
    for (int i = 0; i < t->dims; i++) {
        printf("%d", t->shape[i]);
        if (i < t->dims - 1) {
            printf(", ");
        }
    }
    printf("], size: %d\n", t->size);
    
    printf("Data: ");
    for (int i = 0; i < t->size; i++) {
        printf("%.2f ", t->data[i]);
    }
    printf("\n");
}
void tensor_add(Tensor* t1, Tensor* t2, Tensor* result) {
    if (t1->size != t2->size || t1->size != result->size) {
        fprintf(stderr, "Tensor sizes do not match for addition.\n");
        return;
    }
    for (int i = 0; i < t1->size; i++) {
        result->data[i] = t1->data[i] + t2->data[i];
    }
}
void tensor_subtract(Tensor* t1, Tensor* t2, Tensor* result) {
    if (t1->size != t2->size || t1->size != result->size) {
        fprintf(stderr, "Tensor sizes do not match for subtraction.\n");
        return;
    }
    for (int i = 0; i < t1->size; i++) {
        result->data[i] = t1->data[i] - t2->data[i];
    }
}
void tensor_multiply(Tensor* t1, Tensor* t2, Tensor* result) {
    if (t1->size != t2->size || t1->size != result->size) {
        fprintf(stderr, "Tensor sizes do not match for multiplication.\n");
        return;
    }
    for (int i = 0; i < t1->size; i++) {
        result->data[i] = t1->data[i] * t2->data[i];
    }
}
void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->shape);
    free(t->data);
    free(t);
}

void tensor_dot(Tensor* t1, Tensor* t2, Tensor* result) {
    if (t1->dims != 2 || t2->dims != 2 || t1->shape[1] != t2->shape[0]) {
        fprintf(stderr, "Invalid dimensions for dot product.\n");
        return;
    }
    
    int rows = t1->shape[0];
    int cols = t2->shape[1];
    int inner_dim = t1->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result->data[i * cols + j] = 0.0f;
            for (int k = 0; k < inner_dim; k++) {
                result->data[i * cols + j] += t1->data[i * inner_dim + k] * t2->data[k * cols + j];
            }
        }
    }
}
void tensor_relu(Tensor* t) {
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] < 0) {
            t->data[i] = 0;
        }
    }
}
void tensor_dense_forward(Tensor* input, Tensor* weights, Tensor* bias, Tensor* output) {
    if (input->dims != 2 || weights->dims != 2 || bias->dims != 1 ||
        input->shape[1] != weights->shape[0] || output->shape[0] != input->shape[0] ||
        output->shape[1] != weights->shape[1]) {
        fprintf(stderr, "Invalid dimensions for dense forward.\n");
        return;
    }

    int batch_size = input->shape[0];
    int output_size = weights->shape[1];

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_size; j++) {
            output->data[i * output_size + j] = bias->data[j];
            for (int k = 0; k < weights->shape[0]; k++) {
                output->data[i * output_size + j] += input->data[i * weights->shape[0] + k] * weights->data[k * output_size + j];
            }
        }
    }
}
void cross_entropy_loss(Tensor* logits, Tensor* labels, Tensor* loss) {
    float epsilon = 1e-7f;
    for (int i = 0; i < logits->size; i++) {
        float p = fmaxf(epsilon, fminf(1.0f - epsilon, logits->data[i]));
        loss->data[i] = -labels->data[i] * logf(p);
    }
}

void compute_loss(Tensor* logits, Tensor* labels, Tensor* loss) {
    if (logits->size != labels->size || logits->size != loss->size) {
        fprintf(stderr, "Tensor sizes do not match for loss computation.\n");
        return;
    }
    for (int i = 0; i < logits->size; i++) {
        float diff = logits->data[i] - labels->data[i];
        loss->data[i] = diff * diff; // Mean Squared Error
    }
}
void loss_gradient(Tensor* logits, Tensor* labels, Tensor* gradient) {
    if (logits->size != labels->size || logits->size != gradient->size) {
        fprintf(stderr, "Tensor sizes do not match for loss gradient computation.\n");
        return;
    }
    // For softmax + cross-entropy, gradient is (softmax_output - labels)
    for (int i = 0; i < logits->size; i++) {
        gradient->data[i] = logits->data[i] - labels->data[i];
    }
}
void tensor_softmax(Tensor* t) {
    float max_val = t->data[0];
    for (int i = 1; i < t->size; i++) {
        if (t->data[i] > max_val) max_val = t->data[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        t->data[i] = expf(t->data[i] - max_val);  // stability trick
        sum += t->data[i];
    }

    if (sum < 1e-12f) sum = 1e-12f;  // prevent divide by zero

    for (int i = 0; i < t->size; i++) {
        t->data[i] /= sum;
    }
}



void tensor_subract_scalar(Tensor* t, float scalar) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] -= scalar;
    }
}
void tensor_conv2d(Tensor* input, Tensor* kernel, Tensor* output) {
    if (input->dims != 4 || kernel->dims != 4 || output->dims != 4 ||
        input->shape[1] != kernel->shape[1] || output->shape[0] != input->shape[0] ||
        output->shape[2] != input->shape[2] - kernel->shape[2] + 1 ||
        output->shape[3] != input->shape[3] - kernel->shape[3] + 1) {
        fprintf(stderr, "Invalid dimensions for convolution.\n");
        return;
    }

    int batch_size = input->shape[0];
    int in_channels = input->shape[1];
    int out_channels = kernel->shape[0];
    int kernel_height = kernel->shape[2];
    int kernel_width = kernel->shape[3];
    int output_height = output->shape[2];
    int output_width = output->shape[3];

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                sum += input->data[b * in_channels * input->shape[2] * input->shape[3] +
                                                   ic * input->shape[2] * input->shape[3] +
                                                   (oh + kh) * input->shape[3] + (ow + kw)] *
                                       kernel->data[oc * in_channels * kernel_height * kernel_width +
                                                    ic * kernel_height * kernel_width +
                                                    kh * kernel_width + kw];
                            }
                        }
                    }
                    output->data[b * out_channels * output_height * output_width +
                                  oc * output_height * output_width + oh * output_width + ow] = sum;
                }
            }
        }
    }
}
void tensor_argmax(Tensor* t, int* index) {
    if (t->size == 0) {
        fprintf(stderr, "Tensor is empty, cannot compute argmax.\n");
        return;
    }
    
    float max_value = t->data[0];
    *index = 0;
    
    for (int i = 1; i < t->size; i++) {
        if (t->data[i] > max_value) {
            max_value = t->data[i];
            *index = i;
        }
    }
}
void index_to_char(int index, int* shape, int dims, char* output) {
    for (int i = dims - 1; i >= 0; i--) {
        output[i] = (index % shape[i]) + '0'; // Convert to char
        index /= shape[i];
    }
    output[dims] = '\0'; // Null-terminate the string
}

float tensor_get(Tensor* t, int index) {
    return t->data[index];
}

void tensor_set(Tensor* t, int index, float value) {
    t->data[index] = value;
}

// Add this helper for in-place SGD update
void tensor_subtract_scaled(Tensor* t, Tensor* grad, float lr) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] -= lr * grad->data[i];
    }
}
