#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                m.data[i][j] = 1/(1+expf(-x));
            } else if (a == RELU){
                m.data[i][j] = fmaxf(0,x);
            } else if (a == LRELU){
                if (x <= 0) {
                    m.data[i][j] = .1*x;
                }
            } else if (a == SOFTMAX){
                m.data[i][j] = exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            for(j = 0; j < m.cols; ++j){
                m.data[i][j] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                d.data[i][j] *= x*(1-x);
            } else if (a == RELU){
                if (x <= 0) {
                    d.data[i][j] = 0;
                }
            } else if (a == LRELU){
                if (x <= 0) {
                    d.data[i][j] *= .1;
                }
            }
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation

    // multiply input by weights and apply activation function.
    matrix out = matrix_mult_matrix(in, l->w);
    activate_matrix(out, l->activation);

    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    gradient_matrix(l->out, l->activation, delta);

    // 1.4.2
    free_matrix(l->dw);
    matrix xt = transpose_matrix(l->in);
    matrix dw = matrix_mult_matrix(xt, delta);
    l->dw = dw;
    free_matrix(xt);

    // 1.4.3
    matrix wt = transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta, wt);
    free_matrix(wt);

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // w_{t+1} = w_t + ηΔw_t
    matrix c = axpy_matrix(-decay, l->w, l->dw); //dL/dw_t - λw_t
    matrix wt = axpy_matrix(momentum, l->v, c); //c + m*Δw_{t-1}
    free_matrix(l->v);
    l->v = wt;

    // Update l->w
    matrix w = axpy_matrix(rate, l->v, l->w);
    free_matrix(l->w);
    l->w = w;

    // Remember to free any intermediate results to avoid memory leaks
    free_matrix(c);
}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions
//
// 2.1.1 What are the training and test accuracy values you get? Why might we be interested in both
// training accuracy and testing accuracy? What do these two numbers tell us about our current model?
//
// training accuracy: 90.29 % test accuracy: 90.83 %. We are interested in both values as there will be
// times where the training set with ideal images will have a high accuracy but when it comes to testing,
// we find that we actually only catered to the training set over creating a decent algorithm. The accuracy
// is hight and about the same for both, telling us that we are writing an appropriate algorithm for detecting
// numbers rather than just to pass a certain test.
//
//
// 2.1.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1,
// 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate
// affect both the loss during training and the final model accuracy?
//
// The accuracy between the test and training were around 10% when rates were higher than .1. However,
// there was some increase in accuracy when the rate was .1. The learning rate seemed to decrease
// each time 10 was to a positive power.
//
//
// 2.1.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2,
// 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
//
// Increasing the decay had a lower accuracy overall. Decay under 1 did not affect the values
// too much, but the accuracy was still decreasing there too.
//
//
// 2.2.1 Currently the model uses a logistic activation for the first layer. Try using a the different
// activation functions we programmed. How well do they perform? What's best?
//
// LRELU= training accuracy: 92.24 % test accuracy: 92.43 %
// SOFTMAX= training accuracy: 56.99 % test accuracy: 56.97 %
// LINEAR = training accuracy: 91.34 % test accuracy: 91.58 %
// RELU = training accuracy: 92.42 % test accuracy: 92.61 %
// LOGISTIC= training accuracy: 88.65 % test accuracy: 89.45 %
// RELU had the best training and test accuracy overall, although RELU, LRELU, and LINEAR were fairly
// accurate with about the same amount of accuracy.
//
//
// 2.2.2 Using the same activation, find the best (power of 10) learning rate for your model.
// What is the training accuracy and testing accuracy?
//
// .01 = training accuracy: 92.42 % test accuracy: 92.61 %
// .1 = training accuracy: 95.96 % test accuracy: 95.29 %
// 1 = training accuracy: 20.61 % test accuracy: 20.78 %
// Both training and testing accuracy were better when rate = .1
//
//
// 2.2.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay
// to your model. What happens, does it help? Why or why not may this be?
//
// 0 = training accuracy: 95.96 % test accuracy: 95.29 %
// .1 = training accuracy: 95.56 % test accuracy: 95.10 %
// 1 = training accuracy: 92.85 % test accuracy: 93.00 %
// 10 = training accuracy: 79.02 % test accuracy: 80.00 %
// Both training and testing accuracy were better when decay = .01 where the training accuracy slightly
// decreases with a slightly increased testing accuracy at .1. This may be the case because
// we are analyzing digits (in which there are 10 total), and with us only testing isolated
// digits in a stark black and while background where the amount of digits to guess from
// are limited in number, so our training is appropriate when identifying.
// Additionally, if you slightly increase this decay, our test accuracy increases as well
// (along with a decrease in test as we are enacting more penalty).
//
// 2.2.4 Modify your model so it has 3 layers instead of two. The layers should be
// `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000
// iterations instead of 1000. Look at the training and testing error for different values of
// decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
//
// .0 = training accuracy: 97.92 % test accuracy: 96.70 %
// .00001 = training accuracy: 97.65 % test accuracy: 96.59 %
// .0001 = training accuracy: 97.42 % test accuracy: 96.43 %
// .001 = training accuracy: 97.30 % test accuracy: 96.36 %
// .01 = training accuracy: 97.72 % test accuracy: 96.96 %
// .1 = training accuracy: 97.21 % test accuracy: 96.71 %
// 1 = training accuracy: 92.59 % test accuracy: 92.28 %
// 10 = training accuracy: 76.04 % test accuracy: 76.32 %
// the best decay for training is 0, but the best test accuracy was found at .01.
// As our values get smaller we are likely overfitting towards the training
// (hence the higher training accuracy at 0), although the test accuracy
// is relatively high when the decay is 1 or less than 1. Because a large value
// does not increase our accuracy, however, we can guarantee that we are at least
// not underfitting the data. Because our data is relatively simple (black and white
// digits in an isolated image), our training data is likely adequate enough for our
// tests, making it so that we don't need to have harsher penalties for incorrect guesses.
//
//
// 3.1.1 What is the best training accuracy and testing accuracy? Summarize all the
// hyperparameter combinations you tried.
// rate = .1
// decay = 1
// training accuracy: 15.76 % test accuracy: 15.62 %
//
// rate = .1
// decay = 0
// training accuracy: 18.01 % test accuracy: 17.71 %
//
// rate = .01
// decay = 0
//training accuracy: 47.14 % test accuracy: 45.39 %
// rate = .001
// decay = 0
//training accuracy: 40.63 % test accuracy: 40.49 %
//
// rate = .01
// decay = .01
//training accuracy: 47.06 % test accuracy: 45.19 %
//
// rate = .01
// decay = .1
// training accuracy: 48.45 % test accuracy: 46.62 %
//
// rate = .01
// decay = 1
// training accuracy: 44.34 % test accuracy: 43.89 %
//
// rate = .001
// decay = .1
// training accuracy: 40.40 % test accuracy: 40.49 %

// the best I got was:
// rate = .01
// decay = .1
// training accuracy: 48.45 % test accuracy: 46.62 %

