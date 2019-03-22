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
extern void cpu_func(CByteImage test, int *test_image, std::string testfile, std::string testfilename, std::string trainf, std::string path, int K, int num_classes, int index);
extern void cuda_func(CByteImage test, int *test_image, std::string testfile, std::string testfilename, std::string trainfile, std::string path, int K, int num_classes, int index);

///////////////////////////////////////////////////////////////////////////////
///K-Nearest Neighbours entry point
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{	
	
	CShape shape(IMGSIZE, IMGSIZE, 1);
	CByteImage test(shape);
	std::string path = TEST_PATH;
	int *test_image = (int*)calloc(DIM * DIM, sizeof(int));
	int index = 0;
	
	std::string trainfile = TRAIN;
	std::string testfile = TEST;
	std::string testfilename = TEST_FILE_NAME;

	if (CPU && !GPU)
	{
		printf("CPU Starting: \n");
		auto start = std::chrono::high_resolution_clock::now();
		cpu_func(test, test_image, testfile, testfilename, trainfile, path, KNN, NUMCLASSES, index);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Elapsed time: " << elapsed.count() << " s\n";
		printf("CPU End \n");
	}
	else if (GPU && !CPU)
	{	
		printf("GPU Starting: \n");
		auto start = std::chrono::high_resolution_clock::now();
		cuda_func(test, test_image, testfile, testfilename, trainfile, path, KNN, NUMCLASSES, index);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Elapsed time: " << elapsed.count() << " s\n";
		printf("GPU End \n");
	}
	else
	{	
		printf("CPU Starting: \n");
		auto start = std::chrono::high_resolution_clock::now();
		cpu_func(test, test_image, testfile, testfilename, trainfile, path, KNN, NUMCLASSES, index);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Elapsed time: " << elapsed.count() << " s\n";
		printf("CPU End \n");

		printf("\nGPU Starting: \n");
		start = std::chrono::high_resolution_clock::now();
		cuda_func(test, test_image, testfile, testfilename, trainfile, path, KNN, NUMCLASSES, index);
		finish = std::chrono::high_resolution_clock::now();
		elapsed = finish - start;
		std::cout << "Elapsed time: " << elapsed.count() << " s\n";
		printf("GPU End \n");
	}
	free(test_image);
	return 0;
}
