#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "functional.h"

#define MAX_FILENAME 15


layer *init_node(int num_in, int num_out, char* problem) {
	
	layer* p = malloc(sizeof(layer));

	if (strcmp(problem, "DONUT") == 0) p->batch_size = 9;
	else p->batch_size = 4;

	p->num_in = num_in;
	p->num_out = num_out;

	p->input = malloc(sizeof(float *) * p->batch_size);
	for (int i = 0; i < p->batch_size; i++)
	{
		p->input[i] = malloc(sizeof(float) * p->num_in);
	}
	p->output = malloc(sizeof(float *) * p->batch_size);
	for (int i = 0; i < p->batch_size; i++)
	{
		p->output[i] = malloc(sizeof(float) * p->num_out);
	}	
	p->weight = malloc(sizeof(float *) * p->num_in);
	for (int i = 0; i < p->num_in; i++)
	{
		p->weight[i] = malloc(sizeof(float) * p->num_out);
	}
	p->dinput = malloc(sizeof(float*) * p->batch_size);
	for (int i = 0; i < p->batch_size; i++)
	{
		p->dinput[i] = malloc(sizeof(float) * p->num_in);
	}
	p->doutput = malloc(sizeof(float*) * p->batch_size);
	for (int i = 0; i < p->batch_size; i++)
	{
		p->doutput[i] = malloc(sizeof(float) * p->num_out);
	}
	p->dweight = malloc(sizeof(float*) * p->num_in);
	for (int i = 0; i < p->num_in; i++)
	{
		p->dweight[i] = malloc(sizeof(float) * p->num_out);
	}


	for (int i = 0; i < num_in; i++)
	{
		for (int j = 0; j < num_out; j++)
		{
			p->weight[i][j] = ((float)rand() / (float)RAND_MAX - .5) * 10;
		}
	}

	return p;
}




void layer_forward(layer* floor)
{	
	for (int i = 0; i < floor->batch_size; i++)
	{
		for (int j = 0; j < floor->num_out; j++)
		{
			float net = 0;
			for (int k = 0; k < floor->num_in; k++)
			{
				net += floor->input[i][k] * floor->weight[k][j];
			}
			floor->output[i][j] = sigmoid(net);
		}
	}
}

void model_forward(layer** model, int num_layers)
{
	layer_forward(model[0]);
	for (int i = 1; i < (num_layers - 1); i++)
	{
		for (int j = 0; j < model[i - 1]->batch_size; j++)
		{
			for (int k = 0; k < model[i - 1]->num_out; k++)
			{
				model[i]->input[j][k] = model[i - 1]->output[j][k];
			}
		}
		layer_forward(model[i]);
	}
}




float calculate_loss(layer** model, float target[], int num_layers)
{
	int ind = (num_layers - 1) - 1;
	float loss = 0;

	for (int i = 0; i < model[ind]->batch_size; i++)
	{
		loss += pow(target[i] - model[ind]->output[i][0], 2);
	}
	loss *= .5;

	return loss;
}

float calculate_accuracy(layer** model, float target[], int num_layers)
{
	int ind = (num_layers - 1) - 1;
	float* classified = malloc(sizeof(float) * (model[ind]->batch_size));

	for (int i = 0; i < model[ind]->batch_size; i++)
	{
		if (model[ind]->output[i][0] >= .5) classified[i] = 1.;
		else classified[i] = 0.;
	}
	
	int correct = 0;
	for (int i = 0; i < model[ind]->batch_size; i++)
	{
		if (classified[i] == target[i]) correct++;
	}
	float accuracy = (correct / (float)model[ind]->batch_size);

	return accuracy;
}




void sigmoid_backward(layer** model, float target[], int num_layers)
{
	int ind = (num_layers - 1) - 1;
	for (int i = 0; i < model[ind]->batch_size; i++)
	{
		model[ind]->doutput[i][0] = -(target[i] - model[ind]->output[i][0]);
	}
}


void layer_backward(layer* floor)
{
	// calculate dnet
	float **dnet = malloc(sizeof(float*) * floor->batch_size);
	for (int i = 0; i < floor->batch_size; i++)
	{
		dnet[i] = malloc(sizeof(float) * floor->num_out);
	}

	for (int i = 0; i < floor->batch_size; i++)
	{
		for (int j = 0; j < floor->num_out; j++)
		{
			dnet[i][j] = (floor->doutput[i][j]) * (floor->output[i][j]) * (1 - floor->output[i][j]);
		}
	}

	//calculate dinput
	for (int i = 0; i < floor->batch_size; i++)
	{
		for (int j = 0; j < floor->num_in; j++)
		{
			float value = 0;
			for (int k = 0; k < floor->num_out; k++)
			{
				value += dnet[i][k] * (floor->weight[j][k]);
			}
			floor->dinput[i][j] = value;
		}
	}

	//calculate dweight
	for (int i = 0; i < floor->num_in; i++)
	{
		for (int j = 0; j < floor->num_out; j++)
		{
			float value = 0;
			for (int k = 0; k < floor->batch_size; k++)
			{
				value += (floor->input[k][i]) * dnet[k][j];
			}
			floor->dweight[i][j] = value;
		}
	}
}

void model_backward(layer** model, float target[], int num_layers)
{
	sigmoid_backward(model, target, num_layers);
	int ind = (num_layers - 1) - 1;
	layer_backward(model[ind]);

	for (int i = ind - 1; i >= 0; i--)
	{
		for (int j = 0; j < model[i + 1]->batch_size; j++)
		{
			for (int k = 0; k < model[i + 1]->num_in; k++)
			{
				model[i]->doutput[j][k] = model[i + 1]->dinput[j][k];
			}
		}
		layer_backward(model[i]);
	}
}




void layer_optimize(layer* floor, float learning_rate)
{
	for (int i = 0; i < floor->num_in; i++)
	{
		for (int j = 0; j < floor->num_out; j++)
		{
			floor->weight[i][j] -= learning_rate * floor->dweight[i][j];
		}
	}
}


void model_optimize(layer** model, float learning_rate, int num_layers)
{
	for (int i = 0; i < num_layers - 1; i++)
	{
		layer_optimize(model[i], learning_rate);
	}
}



void record(layer** model, int num_layers, int epoch, float loss, char *problem)
{
	char* weight_file = malloc(sizeof(char) * MAX_FILENAME);
	sprintf(weight_file, "weights_%s.txt", problem);

	char* loss_file = malloc(sizeof(char) * MAX_FILENAME);
	sprintf(loss_file, "loss_%s.txt", problem);

	FILE* weight_fp = fopen(weight_file, "a");
	fprintf(weight_fp, "epoch : %d\n", epoch);

	for (int ind = 0; ind < (num_layers - 1); ind++)
	{
		for (int i = 0; i < model[ind]->num_in; i++)
		{
			for (int j = 0; j < model[ind]->num_out; j++)
			{
				fprintf(weight_fp, "%f ", model[ind]->weight[i][j]);
			}
			fprintf(weight_fp, "\n");
		}
		fprintf(weight_fp, "\n");
	}
	fprintf(weight_fp, "\n");
	fclose(weight_fp);

	FILE* loss_fp = fopen(loss_file, "a");
	fprintf(loss_fp, "%f\n", loss);
	fclose(loss_fp);
}


float sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}