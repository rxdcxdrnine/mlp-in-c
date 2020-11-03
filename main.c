#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <memory.h>

#include "functional.h"
#include "data.h"

#define PROBLEM_CHAR 5


int main(void) {

	srand(time(NULL));
	//input problem
	char* problem = malloc(sizeof(int) * PROBLEM_CHAR);
	printf("Type of Problem (AND/OR/XOR/DONUT) : ");
	scanf("%s", problem);


	//input numner of layers
	int num_layers;
	printf("number of layers : ");
	scanf("%d", &num_layers);


	//input number of nodes per layer (2 nodes in input layer)
	int* num_nodes = malloc(sizeof(int) * num_layers);

	num_nodes[0] = 2;
	printf("number of nodes in input layer : %d\n", num_nodes[0]);
	for (int i = 1; i < (num_layers - 1); i++)
	{
		int num_node;
		printf("number of nodes in hidden layer %d : ", i);
		scanf("%d", &num_node);
		num_nodes[i] = num_node;
	}
	num_nodes[num_layers - 1] = 1;
	printf("number of nodes in output layer : %d\n", num_nodes[num_layers - 1]);

	//input learning rate
	float learning_rate;
	printf("learning rate : ");
	scanf("%f", &learning_rate);


	//get_data
	float **train_set_x = malloc(sizeof(float *) * MAX_BATCH_SIZE);
	for (int i = 0; i < MAX_BATCH_SIZE; i++)
	{
		train_set_x[i] = malloc(sizeof(float) * 2);
	}
	float* train_set_y = malloc(sizeof(float) * MAX_BATCH_SIZE);
	get_data(train_set_x, train_set_y, problem);
	



	//generate model and initialize
	layer** model = malloc(sizeof(layer*) * (num_layers - 1));
	for (int i = 0; i < (num_layers - 1); i++)
	{
		model[i] = init_node(num_nodes[i], num_nodes[i + 1], problem);
	}

	//input data in first layer
	for (int i = 0; i < model[0]->batch_size; i++)
	{
		for (int j = 0; j < model[0]->num_in; j++)
		{
			model[0]->input[i][j] = train_set_x[i][j];
		}
	}

	int epoch = 0;
	while (true) {

		epoch++;

		//forward pass
		model_forward(model, num_layers, num_nodes);

		//calculate loss and accuracy
		float loss = calculate_loss(model, train_set_y, num_layers);
		float accuracy = calculate_accuracy(model, train_set_y, num_layers);

		printf("epoch : %d	loss : %f	accuracy :%f\n", epoch, loss, accuracy);
		if (accuracy == 1.) break;

		//backward propagation
		model_backward(model, train_set_y, num_layers);

		//optimization
		model_optimize(model, learning_rate, num_layers);

		//record weights/loss in file
		record(model, num_layers, epoch, loss, problem);
	}

	// #layers : 5, nodes per hidden layers : 5/5/5, learning_rate : 1 일 때 최적
	// 초기에 더딘 학습 과정을 빠져나가기 위한 random initialization 이 매우 중요
}
