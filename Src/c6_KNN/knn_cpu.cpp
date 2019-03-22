/*
K Nearest Neighbours using CUDA
Authors: Harshul Gupta; Utkarsh Singh
*/

#include <cuda_runtime.h>
#include "common.h"
#include "imageLib/Image.h"
#include "imageLib/ImageIO.h"
#include <helper_functions.h>
#include <iostream>
#include <filesystem>
#include "dirent.h"
#include <sys/types.h>
#include "config.h"
#include <chrono>
#include <vector>

using namespace std;
int *dist_cpu;
int *dist_K;
int *cpu_label;
int *pixel;

/*L1 and L2 Computation on CPU*/
void knn_cpu(int *train_image, int dist_index, int *test_image, int *dist_cpu)
{
	if (METRIC == L1)
	{
		for (int i = 0; i < DIM * DIM; i++)
		{
			pixel[i] = abs(test_image[i] - train_image[i]);
			dist_cpu[dist_index] += pixel[i];
		}
	}
	else if (METRIC == L2)
	{
		for (int i = 0; i < DIM * DIM; i++)
		{
			pixel[i] = pow((test_image[i] - train_image[i]), 2);
			dist_cpu[dist_index] += pixel[i];
		}
		dist_cpu[dist_index] = sqrt(dist_cpu[dist_index]);
	}
}

/*Sorting on CPU*/
void cpu_sort(int K, int index, int lbl, int *dist_cpu, int *dist_K, int *cpu_label)
{
	int i, key, j;
	for (i = 0; i < WSIZE; i++) {
		key = dist_cpu[i];
		j = i - 1;

		while (j >= 0 && dist_cpu[j] > key) {
			dist_cpu[j + 1] = dist_cpu[j];
			j = j - 1;
		}
		dist_cpu[j + 1] = key;
	}
	for (int j = 0; j < K; j++)
	{
		dist_K[j + index] = dist_cpu[j];
		cpu_label[j + index] = lbl;
	}
}

/*CPU classification*/
int cpu_classification(int K, int num_classes, int *dist_K, int *cpu_label)
{
	int i, key, key2, j;
	for (i = 0; i < NUMCLASSES * K; i++) {
		key = dist_K[i];
		key2 = cpu_label[i];
		j = i - 1;

		while (j >= 0 && dist_K[j] > key) {
			dist_K[j + 1] = dist_K[j];
			cpu_label[j + 1] = cpu_label[j];
			j = j - 1;
		}
		dist_K[j + 1] = key;
		cpu_label[j + 1] = key2;
	}

	int *count = (int *)calloc(num_classes, sizeof(int));
	for (int i = 0; i < K; i++) {
		count[cpu_label[i]] += 1;
	}

	int max = 0, f_lbl = 0;
	for (int i = 0; i < num_classes; i++) {
		if (count[i] > max) {
			max = count[i];
			f_lbl = i;
		}
	}
	return f_lbl;
}

/*get image cpu*/
int *get_image_cpu(int *image, CByteImage img) {
	for (int i = 0; i < img.Shape().height; i++) {
		for (int j = 0; j < img.Shape().width; j++) {
			image[img.Shape().height*i + j] = img.Pixel(i, j, 0);
		}
	}
	return image;
}

/*data loader for loading the images and executing knn on cpu*/

void data_loader_cpu(std::string path, int K, int &index, int lbl, int* test_image, int *dist_cpu, int *dist_K, int *cpu_label) {
	CShape shape(IMGSIZE, IMGSIZE, 1);
	CByteImage train(shape);
	int *train_image = (int*)calloc(DIM * DIM, sizeof(int));
	struct dirent *entry;
	DIR *dir = opendir(path.c_str());
	int dist_index = 0;
	while ((entry = readdir(dir)) != NULL) {
		std::string file = entry->d_name;
		const char *dot = strrchr(file.c_str(), '.');
		if (strcmp(dot, ".PNG") == 0 || strcmp(dot, ".png") == 0) {
			std::string myfile = path + "/" + file;
			ReadImageVerb(train, myfile.c_str(), 1);
			train_image = get_image_cpu(train_image, train);
			knn_cpu(train_image, dist_index, test_image, dist_cpu);
			dist_index++;
		}
	}
	free(train_image);
	cpu_sort(K, index, lbl, dist_cpu, dist_K, cpu_label);
	index += K;
}

/*get accuracy for cpu*/

void get_accuracy_cpu(CByteImage test, int *test_image, std::string testf, std::string trainf, std::string path, int K, int num_classes, int index) {
	int correct = 0, total = 0, cpu_prediction = 0;
	std::ifstream testfile(testf);
	std::string   testline;
	std::ifstream trainfile(trainf);
	std::string   trainline;

	while (std::getline(testfile, testline))
	{
		std::stringstream   linestream(testline);
		std::string         filename;
		int                 testlabels;
		std::getline(linestream, filename, ',');
		linestream >> testlabels;
		const char *dot = strrchr(filename.c_str(), '.');
		if (strcmp(dot, ".PNG") == 0 || strcmp(dot, ".png") == 0) {
			std::string myfile = path + "/" + filename;
			ReadImageVerb(test, myfile.c_str(), 1);
			test_image = get_image_cpu(test_image, test);
			index = 0;
			while (std::getline(trainfile, trainline))
			{
				std::stringstream   linestr(trainline);
				std::string         train_path;
				int                 train_labels;
				std::getline(linestr, train_path, ',');
				linestr >> train_labels;
				data_loader_cpu(train_path, K, index, train_labels, test_image, dist_cpu, dist_K, cpu_label);
				memset(dist_cpu, 0, WSIZE * sizeof(int));
			}
			trainfile.clear();
			trainfile.seekg(0, trainfile.beg);
			cpu_prediction = cpu_classification(K, num_classes, dist_K, cpu_label);
			std::string text = "Filename: ";
			text = text + filename + ", True: " + std::to_string(testlabels) + ", Predicted: " + std::to_string(cpu_prediction);
			cout << text << endl;
			if (cpu_prediction == testlabels) correct++;
			total++;
		}
	}
	float accuracy = (float)correct / total;
	printf("Accuracy:%f\n", accuracy);
}

/*get predictions only*/

void get_single_prediction_cpu(CByteImage test, int *test_image, std::string filename, std::string trainf, std::string path, int K, int num_classes, int index) {
	int cpu_prediction = 0;
	std::ifstream trainfile(trainf);
	std::string   trainline;

	const char *dot = strrchr(filename.c_str(), '.');
	if (strcmp(dot, ".PNG") == 0 || strcmp(dot, ".png") == 0) {
		std::string myfile = path + "/" + filename;
		ReadImageVerb(test, myfile.c_str(), 1);
		test_image = get_image_cpu(test_image, test);
		index = 0;
		while (std::getline(trainfile, trainline))
		{
			std::stringstream   linestr(trainline);
			std::string         train_path;
			int                 train_labels;
			std::getline(linestr, train_path, ',');
			linestr >> train_labels;
			data_loader_cpu(train_path, K, index, train_labels, test_image, dist_cpu, dist_K, cpu_label);
			memset(dist_cpu, 0, WSIZE * sizeof(int));
		}
		cpu_prediction = cpu_classification(K, num_classes, dist_K, cpu_label);
		#if(DATASET == MSCD)
			if (cpu_prediction == 1) printf("Prediction is Dog\n");
			else printf("Prediction is Cat\n");
		#elif(DATASET == MNIST)
			printf("Prediction is %d\n", cpu_prediction);
		#endif
	}
}

/*helper function*/

void cpu_func(CByteImage test, int *test_image, std::string testfile, std::string testfilename, std::string trainfile, std::string path, int K, int num_classes, int index)
{
	pixel = (int *)calloc(DIM * DIM, sizeof(int));
	dist_cpu = (int *)calloc(WSIZE, sizeof(int));
	cpu_label = (int *)calloc(num_classes * K, sizeof(int));
	dist_K = (int *)calloc(num_classes * K, sizeof(int));
	if (ACCURACY)
	{	
		get_accuracy_cpu(test, test_image, testfile, trainfile, path, K, num_classes, index);
	}
	else
	{
		get_single_prediction_cpu(test, test_image, testfilename, trainfile, path, K, num_classes, index);
	}
	free(pixel);
	free(dist_cpu);
	free(cpu_label);
	free(dist_K);
}