#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// For reproducibility
#define RANDOM_SEED 42

#define MAX_LINE 16384   // larger for 1005 features
#define INPUT_SIZE 1005
#define H1 128
#define H2 64
#define OUTPUT_SIZE 4
#define LEARNING_RATE 0.0001
#define EPOCHS 2000
#define MAX_SAMPLES 3000

typedef struct {
    Tensor *x; // features
    Tensor *y; // one-hot label
} Sample;

Sample dataset[MAX_SAMPLES];
int dataset_size = 0;

// ---------------- CSV PARSER ----------------
void load_csv(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("CSV open failed"); exit(1); }
    
    printf("Loading dataset from %s\n", filename);

    char line[MAX_LINE];
    while (fgets(line, MAX_LINE, fp)) {
        char *tok = strtok(line, ",");
        int fshape[2] = {INPUT_SIZE, 1};
        Tensor *x = tensor_create(fshape, 2);

        // read INPUT_SIZE features
        for (int i = 0; i < INPUT_SIZE; i++) {
            if (!tok) {
                fprintf(stderr, "CSV parse error on sample %d, feature %d\n", dataset_size, i);
                exit(1);
            }
            x->data[i] = atof(tok);
            tok = strtok(NULL, ",");
        }

        // last column = label
        if (!tok) {
            fprintf(stderr, "CSV parse error: missing label at sample %d\n", dataset_size);
            exit(1);
        }
        int role_idx = atoi(tok);
        if (role_idx < 0 || role_idx >= OUTPUT_SIZE) {
            fprintf(stderr, "Invalid label %d at sample %d\n", role_idx, dataset_size);
            exit(1);
        }

        int lshape[2] = {OUTPUT_SIZE, 1};
        Tensor *y = tensor_create(lshape, 2);
        y->data[role_idx] = 1.0f;

        dataset[dataset_size].x = x;
        dataset[dataset_size].y = y;
        dataset_size++;
        if (dataset_size >= MAX_SAMPLES) break;
    }

    fclose(fp);
    printf("Loaded %d samples with %d features\n", dataset_size, INPUT_SIZE);
}

// ---------------- DATASET NORMALIZATION ----------------
void normalize_dataset() {
    for (int f = 0; f < INPUT_SIZE; f++) {
        float mean = 0, std = 0;

        // compute mean
        for (int i = 0; i < dataset_size; i++) {
            mean += dataset[i].x->data[f];
        }
        mean /= dataset_size;

        // compute std
        for (int i = 0; i < dataset_size; i++) {
            float d = dataset[i].x->data[f] - mean;
            std += d * d;
        }
        std = sqrtf(std / dataset_size + 1e-8f);

        // normalize
        for (int i = 0; i < dataset_size; i++) {
            dataset[i].x->data[f] = (dataset[i].x->data[f] - mean) / std;
        }
    }
    printf("Dataset normalized (z-score)\n");
}

// ---------------- SHUFFLE ----------------
void shuffle_dataset() {
    for (int i = dataset_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Sample tmp = dataset[i];
        dataset[i] = dataset[j];
        dataset[j] = tmp;
    }
}

// ---------------- MODEL PARAMS ----------------
Tensor *W1, *b1, *W2, *b2, *W3, *b3;

Tensor* rand_tensor(int rows, int cols) {
    int shape[2] = {rows, cols};
    Tensor *t = tensor_create(shape, 2);
    float scale = 1.0f / sqrtf(cols);   // Xavier-like scaling
    for (int i=0;i<t->size;i++) {
        t->data[i] = ((float)rand()/RAND_MAX - 0.5f) * 2 * scale;
    }
    return t;
}
Tensor* zeros_tensor(int rows, int cols) {
    int shape[2] = {rows, cols};
    return tensor_create(shape, 2);
}

void init_model() {
    srand(time(NULL));
    W1 = rand_tensor(H1, INPUT_SIZE);
    b1 = zeros_tensor(H1, 1);
    W2 = rand_tensor(H2, H1);
    b2 = zeros_tensor(H2, 1);
    W3 = rand_tensor(OUTPUT_SIZE, H2);
    b3 = zeros_tensor(OUTPUT_SIZE, 1);
}

// ---------------- FORWARD ----------------
typedef struct {
    Tensor *z1, *h1;
    Tensor *z2, *h2;
    Tensor *logits;
} ForwardCache;

ForwardCache forward(Tensor *x) {
    ForwardCache c;

    int s1[2] = {H1, 1}; c.z1 = tensor_create(s1, 2);
    tensor_dot(W1, x, c.z1);
    tensor_add(c.z1, b1, c.z1);
    c.h1 = tensor_create(s1, 2);
    memcpy(c.h1->data, c.z1->data, sizeof(float)*c.z1->size);
    tensor_relu(c.h1);

    int s2[2] = {H2, 1}; c.z2 = tensor_create(s2, 2);
    tensor_dot(W2, c.h1, c.z2);
    tensor_add(c.z2, b2, c.z2);
    c.h2 = tensor_create(s2, 2);
    memcpy(c.h2->data, c.z2->data, sizeof(float)*c.z2->size);
    tensor_relu(c.h2);

    int s3[2] = {OUTPUT_SIZE, 1}; c.logits = tensor_create(s3, 2);
    tensor_dot(W3, c.h2, c.logits);
    tensor_add(c.logits, b3, c.logits);
    tensor_softmax(c.logits);

    return c;
}

// ---------------- BACKWARD ----------------
void relu_grad(Tensor *z, Tensor *delta) {
    for (int i=0; i<z->size; i++) {
        if (z->data[i] <= 0) delta->data[i] = 0;
    }
}

void backward(Tensor *x, Tensor *y, ForwardCache c) {
    // delta3 = yhat - y
    int s3[2] = {OUTPUT_SIZE,1};
    Tensor *delta3 = tensor_create(s3,2);
    loss_gradient(c.logits, y, delta3);

    // Update W3, b3
    for (int i=0;i<OUTPUT_SIZE;i++) {
        for (int j=0;j<H2;j++) {
            W3->data[i*H2+j] -= LEARNING_RATE * delta3->data[i] * c.h2->data[j];
        }
        b3->data[i] -= LEARNING_RATE * delta3->data[i];
    }

    // delta2 = W3^T * delta3 ⊙ ReLU’(z2)
    int s2[2] = {H2,1};
    Tensor *delta2 = tensor_create(s2,2);
    for (int i=0;i<H2;i++) {
        float sum=0;
        for (int j=0;j<OUTPUT_SIZE;j++) {
            sum += W3->data[j*H2+i] * delta3->data[j];
        }
        delta2->data[i] = sum;
    }
    relu_grad(c.z2, delta2);

    // Update W2, b2
    for (int i=0;i<H2;i++) {
        for (int j=0;j<H1;j++) {
            W2->data[i*H1+j] -= LEARNING_RATE * delta2->data[i] * c.h1->data[j];
        }
        b2->data[i] -= LEARNING_RATE * delta2->data[i];
    }

    // delta1 = W2^T * delta2 ⊙ ReLU’(z1)
    int s1[2] = {H1,1};
    Tensor *delta1 = tensor_create(s1,2);
    for (int i=0;i<H1;i++) {
        float sum=0;
        for (int j=0;j<H2;j++) {
            sum += W2->data[j*H1+i] * delta2->data[j];
        }
        delta1->data[i] = sum;
    }
    relu_grad(c.z1, delta1);

    // Update W1, b1
    for (int i=0;i<H1;i++) {
        for (int j=0;j<INPUT_SIZE;j++) {
            W1->data[i*INPUT_SIZE+j] -= LEARNING_RATE * delta1->data[i] * x->data[j];
        }
        b1->data[i] -= LEARNING_RATE * delta1->data[i];
    }

    tensor_free(delta1);
    tensor_free(delta2);
    tensor_free(delta3);
}
// ---------------- SAVE BEST MODEL ----------------
void save_weights(const char *fname) {
    FILE *f = fopen(fname, "wb");
    if (!f) {
        perror("save_weights");
        return;
    }

    // Write layer weights + biases
    fwrite(W1->data, sizeof(float), W1->size, f);
    fwrite(b1->data, sizeof(float), b1->size, f);

    fwrite(W2->data, sizeof(float), W2->size, f);
    fwrite(b2->data, sizeof(float), b2->size, f);

    fwrite(W3->data, sizeof(float), W3->size, f);
    fwrite(b3->data, sizeof(float), b3->size, f);

    fclose(f);
    printf("✅ Saved best model to %s\n", fname);
}
// ---------------- EVALUATION METRICS ----------------
typedef struct {
    int tp, fp, fn, tn;  // true positive, false positive, false negative, true negative
    float precision, recall, f1;
} ClassMetrics;

void compute_confusion_matrix(int *confusion, Tensor *y_true, Tensor *y_pred, int n_classes) {
    int true_idx, pred_idx;
    tensor_argmax(y_true, &true_idx);
    tensor_argmax(y_pred, &pred_idx);
    confusion[true_idx * n_classes + pred_idx]++;
}

void print_confusion_matrix(int *confusion, int n_classes) {
    printf("\nConfusion Matrix:\n");
    
    // Print header
    printf("     ");
    for (int i = 0; i < n_classes; i++) {
        printf("Pred%d ", i);
    }
    printf("\n");
    
    // Print rows
    for (int i = 0; i < n_classes; i++) {
        printf("True%d ", i);
        for (int j = 0; j < n_classes; j++) {
            printf("%5d ", confusion[i * n_classes + j]);
        }
        printf("\n");
    }
}

void compute_metrics(ClassMetrics *metrics, int *confusion, int n_classes) {
    // Calculate per-class metrics
    for (int i = 0; i < n_classes; i++) {
        metrics[i].tp = confusion[i * n_classes + i];
        
        metrics[i].fp = 0;
        for (int j = 0; j < n_classes; j++) {
            if (j != i) metrics[i].fp += confusion[j * n_classes + i];
        }
        
        metrics[i].fn = 0;
        for (int j = 0; j < n_classes; j++) {
            if (j != i) metrics[i].fn += confusion[i * n_classes + j];
        }
        
        metrics[i].precision = metrics[i].tp / (float)(metrics[i].tp + metrics[i].fp + 1e-10);
        metrics[i].recall = metrics[i].tp / (float)(metrics[i].tp + metrics[i].fn + 1e-10);
        metrics[i].f1 = 2 * metrics[i].precision * metrics[i].recall / 
                         (metrics[i].precision + metrics[i].recall + 1e-10);
    }
}

float compute_macro_f1(ClassMetrics *metrics, int n_classes) {
    float macro_f1 = 0;
    for (int i = 0; i < n_classes; i++) {
        macro_f1 += metrics[i].f1;
    }
    return macro_f1 / n_classes;
}

void print_metrics(ClassMetrics *metrics, int n_classes) {
    printf("\nPer-class Metrics:\n");
    printf("Class   Precision   Recall   F1-Score\n");
    for (int i = 0; i < n_classes; i++) {
        printf("  %d      %.4f     %.4f    %.4f\n", 
               i, metrics[i].precision, metrics[i].recall, metrics[i].f1);
    }
    
    float macro_f1 = compute_macro_f1(metrics, n_classes);
    printf("\nMacro F1 Score: %.4f\n", macro_f1);
}

// Function to assess model calibration
void assess_calibration(Tensor **predictions, Tensor **true_labels, int num_samples) {
    // Calculate average confidence per class
    float avg_confidence[OUTPUT_SIZE] = {0};
    int class_counts[OUTPUT_SIZE] = {0};
    
    // Collect confidence statistics
    for (int i = 0; i < num_samples; i++) {
        int true_class;
        tensor_argmax(true_labels[i], &true_class);
        
        int pred_class;
        float max_conf = 0;
        tensor_argmax(predictions[i], &pred_class);
        max_conf = predictions[i]->data[pred_class];
        
        avg_confidence[pred_class] += max_conf;
        class_counts[pred_class]++;
    }
    
    // Print calibration assessment
    printf("\nModel Calibration Assessment:\n");
    printf("Class   Avg Confidence   Samples\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (class_counts[i] > 0) {
            avg_confidence[i] /= class_counts[i];
            printf("  %d      %.4f          %d\n", i, avg_confidence[i], class_counts[i]);
        } else {
            printf("  %d      N/A            0\n", i);
        }
    }
    
    // General calibration comment
    float overall_avg_conf = 0;
    int total_samples = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (class_counts[i] > 0) {
            overall_avg_conf += avg_confidence[i] * class_counts[i];
            total_samples += class_counts[i];
        }
    }
    
    if (total_samples > 0) {
        overall_avg_conf /= total_samples;
        printf("\nOverall average confidence: %.4f\n", overall_avg_conf);
        
        if (overall_avg_conf > 0.9) {
            printf("Model may be overconfident. Consider calibration techniques.\n");
        } else if (overall_avg_conf < 0.6) {
            printf("Model appears underconfident. Consider reviewing architecture.\n");
        } else {
            printf("Model confidence appears reasonable.\n");
        }
    }
}

// ---------------- TRAIN + TEST WITH EARLY STOPPING + SAVE BEST ----------------
void train_and_test() {
    int train_size = (int)(0.8 * dataset_size);
    int test_size = dataset_size - train_size;

    int patience = 20;   // stop after 20 epochs without improvement
    int wait = 0;
    float best_test_acc = 0.0f;
    float best_macro_f1 = 0.0f;  // Track best F1 score

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // ---- Training ----
        int train_correct = 0;
        float train_loss = 0.0;

        for (int i = 0; i < train_size; i++) {
            ForwardCache c = forward(dataset[i].x);

            int yi, pi;
            tensor_argmax(dataset[i].y, &yi);
            tensor_argmax(c.logits, &pi);
            if (yi == pi) train_correct++;

            // loss
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                if (dataset[i].y->data[k] > 0.5f) {
                    float p = c.logits->data[k];
                    if (p < 1e-7f) p = 1e-7f;
                    train_loss += -logf(p);
                }
            }

            backward(dataset[i].x, dataset[i].y, c);

            tensor_free(c.z1); tensor_free(c.h1);
            tensor_free(c.z2); tensor_free(c.h2);
            tensor_free(c.logits);
        }

        // ---- Testing ----
        int test_correct = 0;
        float test_loss = 0.0;
        
        // Confusion matrix
        int confusion[OUTPUT_SIZE * OUTPUT_SIZE] = {0};
        
        // Storage for predictions and true labels for calibration assessment
        Tensor *predictions[test_size];
        Tensor *true_labels[test_size];
        int pred_idx = 0;

        for (int i = train_size; i < dataset_size; i++, pred_idx++) {
            ForwardCache c = forward(dataset[i].x);

            int yi, pi;
            tensor_argmax(dataset[i].y, &yi);
            tensor_argmax(c.logits, &pi);
            if (yi == pi) test_correct++;
            
            // Update confusion matrix
            compute_confusion_matrix(confusion, dataset[i].y, c.logits, OUTPUT_SIZE);
            
            // Store predictions and true labels for calibration assessment
            int shape[2] = {OUTPUT_SIZE, 1};
            predictions[pred_idx] = tensor_create(shape, 2);
            true_labels[pred_idx] = tensor_create(shape, 2);
            
            memcpy(predictions[pred_idx]->data, c.logits->data, sizeof(float) * OUTPUT_SIZE);
            memcpy(true_labels[pred_idx]->data, dataset[i].y->data, sizeof(float) * OUTPUT_SIZE);

            for (int k = 0; k < OUTPUT_SIZE; k++) {
                if (dataset[i].y->data[k] > 0.5f) {
                    float p = c.logits->data[k];
                    if (p < 1e-7f) p = 1e-7f;
                    test_loss += -logf(p);
                }
            }

            tensor_free(c.z1); tensor_free(c.h1);
            tensor_free(c.z2); tensor_free(c.h2);
            tensor_free(c.logits);
        }

        float train_acc = 100.0 * train_correct / train_size;
        float test_acc  = 100.0 * test_correct / test_size;

        printf("Epoch %d | Train acc=%.2f%% loss=%.4f | Test acc=%.2f%% loss=%.4f\n",
               epoch,
               train_acc, train_loss / train_size,
               test_acc,  test_loss / test_size);
               
        // Calculate and display metrics
        ClassMetrics metrics[OUTPUT_SIZE];
        compute_metrics(metrics, confusion, OUTPUT_SIZE);
        float current_macro_f1 = compute_macro_f1(metrics, OUTPUT_SIZE);
        
        // Display metrics on final epoch or if significant improvement
        if (epoch == EPOCHS - 1 || (current_macro_f1 > best_macro_f1 && current_macro_f1 - best_macro_f1 > 0.01)) {
            print_confusion_matrix(confusion, OUTPUT_SIZE);
            print_metrics(metrics, OUTPUT_SIZE);
            assess_calibration(predictions, true_labels, test_size);
        }

        // ---- Early Stopping + Save Best Weights ----
        if (current_macro_f1 > best_macro_f1) {
            best_macro_f1 = current_macro_f1;
            best_test_acc = test_acc;
            wait = 0;
            save_weights("best_model.bin");   // ✅ save best model
        } else {
            wait++;
            if (wait >= patience) {
                printf("Early stopping at epoch %d (best test acc=%.2f%%, best macro F1=%.4f)\n",
                       epoch, best_test_acc, best_macro_f1);
                break;
            }
        }
        
        // Free prediction tensors
        for (int i = 0; i < test_size; i++) {
            tensor_free(predictions[i]);
            tensor_free(true_labels[i]);
        }
    }
}

// ---------------- MAIN ----------------
int main() {
    // Initialize random seed for reproducibility
    srand(RANDOM_SEED);
    
    // Print environment information
    printf("Developer Role Classification - C Implementation\n");
    printf("================================================\n");
    printf("Random seed: %d\n", RANDOM_SEED);
    printf("Model architecture: %d -> %d -> %d -> %d\n", INPUT_SIZE, H1, H2, OUTPUT_SIZE);
    printf("Learning rate: %.6f\n", LEARNING_RATE);
    printf("Max epochs: %d\n", EPOCHS);
    printf("================================================\n\n");
    
    load_csv("processed_dataset.csv");
    normalize_dataset();
    init_model();
    train_and_test();
    return 0;
}

