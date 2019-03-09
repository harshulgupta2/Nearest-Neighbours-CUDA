/*
K Nearest Neighbours using CUDA
Authors: Harshul Gupta; Utkarsh Singh

 */

extern void knn_cuda(int * ref, int *test);

#include <cuda_runtime.h>

#include "common.h"
#include "imageLib/Image.h"
#include "imageLib/ImageIO.h"
#include <helper_functions.h>
#include "imageLib/Convert.h"

#include <iostream>
#include <filesystem>
#include "dirent.h"
#include <sys/types.h>
using namespace std;

int *get_image(int *image, CByteImage img) {
	img = ConvertToGray(img);
	for (int i = 0; i < img.Shape().height; i++) {
		for (int j = 0; j < img.Shape().width; j++) {
			image[img.Shape().height*i + j] = img.Pixel(i, j, 0);
		}
	}
	return image;
}

///////////////////////////////////////////////////////////////////////////////
/// application entry point
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	std::string path = "C:/Users/h6gupta/Downloads/dataset";
	struct dirent *entry;
	DIR *dir = opendir(path.c_str());
	while ((entry = readdir(dir)) != NULL) {
		std::string file = entry->d_name;
		// Determine the file extension
		const char *dot = strrchr(file.c_str(), '.');
		if (strcmp(dot, ".PNG") == 0 || strcmp(dot, ".png") == 0) {
			std::string myfile = path + "/" + file;
			cout << myfile << endl;
			//CByteImage train;
			//ReadImageVerb(train, myfile.c_str(), 1);
			//WriteImageVerb(train, outfile, 1);
			//break;
		}
	}

	CByteImage train, test;
	char *f1 = "C:/Users/h6gupta/Downloads/train.png";
	char *f2 = "C:/Users/h6gupta/Downloads/test.png";

	ReadImageVerb(train, f1, 1);
	ReadImageVerb(test, f2, 1);

	train = ConvertToGray(train);
	test = ConvertToGray(test);

	int *train_image = (int*)malloc(train.Shape().width * train.Shape().height * sizeof(int));
	train_image = get_image(train_image, train);

	int *test_image = (int*)malloc(test.Shape().width * test.Shape().height * sizeof(int));
	test_image = get_image(test_image, test);

	//printf("%d, %d\t", test_image[256 + 255],test.Pixel(1, 255, 0));

	//printf("%d", test_image[-1]);

	knn_cuda(train_image, test_image);
	return 0;
	//printf("%d\t%d", train_image[0], test_image[0]);
	//printf("\n%d\t%d", train.Pixel(0,0,0), test.Pixel(0, 0, 0));


	/*Subraction*/

	//Perform_Difference<<< >>>




	//WriteImageVerb(img, outfile, 1);
	
	/*
    // welcome message
    printf("%s Starting...\n\n", sSDKsample);

    // pick GPU
    findCudaDevice(argc, (const char **)argv);

    // find images
    const char *const sourceFrameName = "frame10.ppm";
    const char *const targetFrameName = "frame11.ppm";

    // image dimensions
    int width;
    int height;
    // row access stride
    int stride;

    // flow is computed from source image to target image
    float *h_source; // source image, host memory
    float *h_target; // target image, host memory

    // load image from file
    if (!LoadImageAsFP32(h_source, width, height, stride, sourceFrameName, argv[0]))
    {
        exit(EXIT_FAILURE);
    }

    if (!LoadImageAsFP32(h_target, width, height, stride, targetFrameName, argv[0]))
    {
        exit(EXIT_FAILURE);
    }

    // allocate host memory for CPU results
    float *h_uGold = new float [stride * height];
    float *h_vGold = new float [stride * height];

    // allocate host memory for GPU results
    float *h_u   = new float [stride * height];
    float *h_v   = new float [stride * height];

    // smoothness
    // if image brightness is not within [0,1]
    // this paramter should be scaled appropriately
    const float alpha = 0.2f;

    // number of pyramid levels
    const int nLevels = 5;

    // number of solver iterations on each level
    const int nSolverIters = 500;

    // number of warping iterations
    const int nWarpIters = 3;

    ComputeFlowGold(h_source, h_target, width, height, stride, alpha,
                    nLevels, nWarpIters, nSolverIters, h_uGold, h_vGold);

    ComputeFlowCUDA(h_source, h_target, width, height, stride, alpha,
                    nLevels, nWarpIters, nSolverIters, h_u, h_v);

    // compare results (L1 norm)
    bool
    status = CompareWithGold(width, height, stride, h_uGold, h_vGold, h_u, h_v);

    WriteFloFile("FlowGPU.flo", width, height, stride, h_u, h_v);

    WriteFloFile("FlowCPU.flo", width, height, stride, h_uGold, h_vGold);

	CFloatImage im;
	float maxmotion = -1;
	ReadFlowFile(im, "FlowGPU.flo");
	CByteImage band, outim;
	CShape sh = im.Shape();
	sh.nBands = 3;
	outim.ReAllocate(sh);
	outim.ClearPixels();
	MotionToColor(im, outim, maxmotion);
	WriteImageVerb(outim, "FlowGPU.png", verbose);

    // free resources
    delete [] h_uGold;
    delete [] h_vGold;

    delete [] h_u;
    delete [] h_v;

    delete [] h_source;
    delete [] h_target;

    // report self-test status
    exit(status ? EXIT_SUCCESS : EXIT_FAILURE);

	*/
}
