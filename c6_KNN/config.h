/*
K Nearest Neighbours using CUDA
Authors: Harshul Gupta; Utkarsh Singh
*/

#define LOWER_BIT 0
#define KNN 3
#define L1 1
#define L2 0
#define MSCD 1
#define MNIST 2
#define WSIZE 32
#define ACCURACY 1
#define CPU 1
#define GPU 1
#define RADIX 0
#define INSERTION 1
#define METRIC L2
#define DATASET MSCD
#define SORTING RADIX

#if DATASET == MSCD
#define DIM 256
#define IMGSIZE 256
#define NUMCLASSES 2
#define SHARED 128
#define THREADS 128
#define SUM_BLK_SIZE 128
#define SUM_OUT (DIM*DIM/SUM_BLK_SIZE)
#define TRAIN "../../../../MSCD/train.txt"
#define TEST "../../../../MSCD/labels.txt"
#define TEST_PATH "../../../../MSCD/test"
#define TEST_FILE_NAME "dog4.png"

#elif DATASET == MNIST
#define DIM 32
#define IMGSIZE 28
#define NUMCLASSES 10
#define SHARED 128
#define THREADS 128
#define SUM_BLK_SIZE 32
#define SUM_OUT (DIM*DIM/SUM_BLK_SIZE)
#define TRAIN "../../../../MNIST/train.txt"
#define TEST "../../../../MNIST/labels.txt"
#define TEST_PATH "../../../../MNIST/testing_dataset"
#define TEST_FILE_NAME "30.png"
#endif


#if(METRIC == L1)
#define UPPER_BIT 24
#else
#define UPPER_BIT 32
#endif


