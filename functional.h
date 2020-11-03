#include <stdio.h>
#include <math.h>

typedef struct {
	int batch_size;
	int num_in;
	int num_out;

	float** input;
	float** output;
	float** weight;

	float** dinput;
	float** doutput;
	float** dweight;
} layer;

layer* init_node(int num_in, int num_out, char* problem);

void layer_forward(layer* floor);
void model_forward(layer** model, int num_layers);

float calculate_loss(layer** model, float target[], int num_layers);
float calculate_accuracy(layer** model, float target[], int num_layers);

void sigmoid_backward(layer** model, float target[], int num_layers);
void layer_backward(layer* floor);
void model_backward(layer** model, float target[], int num_layers);

void layer_optimize(layer* floor, float learning_rate);
void model_optimize(layer** model, float learning_rate, int num_layers);

void record(layer** model, int num_layers, int epoch, float loss, char *problem);

float sigmoid(float x);