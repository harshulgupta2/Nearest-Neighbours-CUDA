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
extern void knn_cuda(int *train_image, int dist_index);
extern void knn_sort(int K, int &index, int lbl);
extern void knn_init(int K);
extern void transfer_testimage(int *test_image);
extern int perform_classification(int K, int num_classes);
extern void cuda_deallocation();

/*convert image to an array*/

int *get_image_gpu(int *image, CByteImage img) {
	for (int i = 0; i < img.Shape().height; i++) {
		for (int j = 0; j < img.Shape().width; j++) {
			image[img.Shape().height*i + j] = img.Pixel(i, j, 0);
		}
	}
	return image;
}

/*data loader for loading the images and executing knn on gpu*/

void data_loader_gpu(std::string path, int K, int &index, int lbl) {
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
			train_image = get_image_gpu(train_image, train);
			knn_cuda(train_image, dist_index);
			dist_index++;
		}
	}
	free(train_image);
	knn_sort(K, index, lbl);
	index += K;
}

/*get accuracy for gpu*/

void get_accuracy_gpu(CByteImage test, int *test_image, std::string testf, std::string trainf, std::string path, int K, int num_classes, int index) {
	int correct = 0, total = 0, prediction = 0;
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
			test_image = get_image_gpu(test_image, test);
			transfer_testimage(test_image);
			index = 0;
			while (std::getline(trainfile, trainline))
			{
				std::stringstream   linestr(trainline);
				std::string         train_path;
				int                 train_labels;
				std::getline(linestr, train_path, ',');
				linestr >> train_labels;
				data_loader_gpu(train_path, K, index, train_labels);
			}
			trainfile.clear();
			trainfile.seekg(0, trainfile.beg);
			prediction = perform_classification(K, num_classes);
			std::string text = "Filename: ";
			text = text + filename + ", True: " + std::to_string(testlabels) + ", Predicted: " + std::to_string(prediction);
			cout << text << endl;
			if (prediction == testlabels) correct++;
			total++;
		}
	}
	float accuracy = (float)correct / total;
	printf("Accuracy:%f\n", accuracy);
}

/*get predictions only*/

void get_single_prediction_gpu(CByteImage test, int *test_image, std::string filename, std::string trainf, std::string path, int K, int num_classes, int index) {
	int prediction = 0;
	std::ifstream trainfile(trainf);
	std::string   trainline;

	const char *dot = strrchr(filename.c_str(), '.');
	if (strcmp(dot, ".PNG") == 0 || strcmp(dot, ".png") == 0) {
		std::string myfile = path + "/" + filename;
		ReadImageVerb(test, myfile.c_str(), 1);
		test_image = get_image_gpu(test_image, test);
		transfer_testimage(test_image);
		index = 0;
		while (std::getline(trainfile, trainline))
		{
			std::stringstream   linestr(trainline);
			std::string         train_path;
			int                 train_labels;
			std::getline(linestr, train_path, ',');
			linestr >> train_labels;
			data_loader_gpu(train_path, K, index, train_labels);
		}
		prediction = perform_classification(K, num_classes);
		#if(DATASET == MSCD)
			if (prediction == 1) printf("Prediction is Dog\n");
			else printf("Prediction is Cat\n");
		#elif(DATASET == MNIST)
			printf("Prediction is %d\n", prediction);
		#endif
	}
}

/*helper function*/

void cuda_func(CByteImage test, int *test_image, std::string testfile, std::string testfilename, std::string trainfile, std::string path, int K, int num_classes, int index)
{
	knn_init(KNN);
	if (ACCURACY)
	{
		get_accuracy_gpu(test, test_image, testfile, trainfile, path, K, num_classes, index);
	}
	else
	{
		get_single_prediction_gpu(test, test_image, testfilename, trainfile, path, K, num_classes, index);
	}
	cuda_deallocation();
}