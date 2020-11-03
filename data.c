#include <stdio.h>

#include "data.h"

#define OP_BATCH_SIZE 4
#define MAX_BATCH_SIZE 9


float AND_train_set_x[][2] = {
	{0., 0.},
	{0., 1.},
	{1., 0.},
	{1., 1.},
};
float AND_train_set_y[] = { 0, 0, 0, 1 };

float OR_train_set_x[][2] = {
	{0., 0.},
	{0., 1.},
	{1., 0.},
	{1., 1.},
};
float OR_train_set_y[] = { 0, 0, 0, 1 };

float XOR_train_set_x[][2] = {
	{0., 0.},
	{0., 1.},
	{1., 0.},
	{1., 1.},
};
float XOR_train_set_y[] = { 0, 1, 1, 0 };

float DONUT_train_set_x[][2] = {
	{0., 0.},
	{0., 1.},
	{1., 0.},
	{1. ,1.},
	{.5, 1.},
	{1., .5},
	{0., .5},
	{.5, 0.},
	{.5, .5},
};
float DONUT_train_set_y[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };



void get_data(float **train_set_x, float *train_set_y, char *problem)
{
	if (strcmp(problem, "AND") == 0)
	{
		for (int i = 0; i < OP_BATCH_SIZE; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				train_set_x[i][j] = AND_train_set_x[i][j];
			}
		}
		for (int i = 0; i < OP_BATCH_SIZE; i++)
		{
			train_set_y[i] = AND_train_set_y[i];
		}
	}
	else if (strcmp(problem, "OR") == 0)
	{
		for (int i = 0; i < OP_BATCH_SIZE; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				train_set_x[i][j] = OR_train_set_x[i][j];
			}
		}
		for (int i = 0; i < OP_BATCH_SIZE; i++)
		{
			train_set_y[i] = OR_train_set_y[i];
		}
	}
	else if (strcmp(problem, "XOR") == 0)
	{
		for (int i = 0; i < OP_BATCH_SIZE; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				train_set_x[i][j] = XOR_train_set_x[i][j];
			}
		}
		for (int i = 0; i < OP_BATCH_SIZE; i++)
		{
			train_set_y[i] = XOR_train_set_y[i];
		}
	}
	else
	{
		for (int i = 0; i < MAX_BATCH_SIZE; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				train_set_x[i][j] = DONUT_train_set_x[i][j];
			}
		}
		for (int i = 0; i < MAX_BATCH_SIZE; i++)
		{
			train_set_y[i] = DONUT_train_set_y[i];
		}
	}

}
