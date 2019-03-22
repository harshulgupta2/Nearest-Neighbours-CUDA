/*
K Nearest Neighbours using CUDA
Authors: Harshul Gupta; Utkarsh Singh
*/

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include<cmath>
#include<float.h>
#include "config.h"
#include <chrono> 

int* d_train;
int* d_test;
int* pixelnorm;
int* sum;
int *d_norm_array;
int *d_k_norms;
int *d_labels;
int dim = DIM;
int *d_odata;
int *h_odata = (int*)malloc(sizeof(int) * SUM_OUT);
int *norm_array = (int*)malloc(sizeof(int) * WSIZE);

__shared__ int s_sum[SUM_BLK_SIZE];
__shared__ int radix[WSIZE];
__shared__ int p_dist[WSIZE];

/*Initialize CUDA memories*/

void knn_init(int K)
{
	checkCudaErrors(cudaMalloc((void **)&d_train, (sizeof(int) * DIM * DIM)));
	checkCudaErrors(cudaMalloc((void **)&d_test, (sizeof(int) * DIM * DIM)));
	checkCudaErrors(cudaMalloc((void **)&pixelnorm, (sizeof(int) * DIM * DIM)));
	checkCudaErrors(cudaMalloc((void **)&sum, (sizeof(int) * 2)));
	checkCudaErrors(cudaMalloc((void **)&d_norm_array, (sizeof(int) * WSIZE)));
	checkCudaErrors(cudaMalloc((void **)&d_k_norms, (sizeof(int) * NUMCLASSES * K)));
	checkCudaErrors(cudaMalloc((void **)&d_labels, (sizeof(int) * NUMCLASSES * K)));
	checkCudaErrors(cudaMalloc((void **)&d_odata, (sizeof(int) * SUM_OUT)));
}

/*one time transfer of test image to GPU*/

void transfer_testimage(int *test_image) {
	checkCudaErrors(cudaMemcpy(d_test, test_image, ((sizeof(int)) * DIM * DIM), cudaMemcpyHostToDevice));
}

/*Radix Sort Unrolled by 2*/
__global__ void RadixSort(int *arr, int nx)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int value, zeros, scan, final_sum, temp, out;
	radix[idx] = arr[idx];
	if (idx < nx)
	{
		for (int bit = LOWER_BIT; bit < UPPER_BIT; bit+=2) {
			value = radix[idx];
			zeros = __ballot(!((value >> bit) & 0x1));
			scan = __popc(zeros&((1 << idx) - 1));
			final_sum = (zeros >> (nx - 1) & 0x1) + __shfl(scan, (nx - 1));
			temp = (idx - scan + final_sum);
			out = ((value >> bit) & 0x1) ? temp : scan;
			radix[out] = value;

			value = radix[idx];
			zeros = __ballot(!((value >> (bit+1)) & 0x1));
			scan = __popc(zeros&((1 << idx) - 1));
			final_sum = (zeros >> (nx - 1) & 0x1) + __shfl(scan, (nx - 1));
			temp = (idx - scan + final_sum);
			out = ((value >> (bit+1)) & 0x1) ? temp : scan;
			radix[out] = value;

		}
	}
	arr[idx] = radix[idx];
}

__global__ void RadixSortLabels(int *arr, int *labels, int nx)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int value, zeros, scan, final_sum, temp, out, lbl;
	if (idx < nx)
	{
		for (int bit = LOWER_BIT; bit < UPPER_BIT; bit++) {
			value = arr[idx];
			lbl = labels[idx];
			zeros = __ballot(!((value >> bit) & 0x1));
			scan = __popc(zeros&((1 << idx) - 1));
			final_sum = (zeros >> (nx - 1) & 0x1) + __shfl(scan, (nx - 1));
			temp = (idx - scan + final_sum);
			out = ((value >> (bit)) & 0x1) ? temp : scan;
			arr[out] = value;
			labels[out] = lbl;
		}
	}
}

/*Insertion Sort for Comparison Purposes*/
__global__ void InsertionSort(int* arr, int nx) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nx) {
		p_dist[idx] = arr[idx];
		#pragma unroll
		for (int i = 1; i<nx; i++) {
			int curr_dist = p_dist[i];
			int j = i-1;
			while (j >= 0 && p_dist[(j)] > curr_dist) {
				p_dist[j+1] = p_dist[(j)];
				--j;
			}
			p_dist[j+1] = curr_dist;
		}
	}
	arr[idx] = p_dist[idx];
}

/*Calculate L1 distance*/
__global__ void knn_distance_L1(int *d_train, int *d_test, int *pixelnorm, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;
	if (ix < nx*ny) {
		pixelnorm[ix] = fabsf(d_test[ix] - d_train[ix]);
	}
}

/*Calculate L2 distance*/
__global__ void knn_distance_L2(int *d_train, int *d_test, int *pixelnorm, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;
	pixelnorm[ix] = (d_test[ix] - d_train[ix]) * (d_test[ix] - d_train[ix]);
}

/*Calculate Warp Level Sum*/
template <unsigned int iBlockSize>
__global__ void WarpSumUnroll8(int *pixels, int *output, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
	int localSum = 0;

	if (idx + 7 * blockDim.x < n)
	{
		int a1 = pixels[idx];
		int a2 = pixels[idx + blockDim.x];
		int a3 = pixels[idx + 2 * blockDim.x];
		int a4 = pixels[idx + 3 * blockDim.x];
		int b1 = pixels[idx + 4 * blockDim.x];
		int b2 = pixels[idx + 5 * blockDim.x];
		int b3 = pixels[idx + 6 * blockDim.x];
		int b4 = pixels[idx + 7 * blockDim.x];
		localSum = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}

	s_sum[tid] = localSum;
	__syncthreads();

	if (iBlockSize >= 1024 && tid < 512) s_sum[tid] += s_sum[tid + 512];
	__syncthreads();
	if (iBlockSize >= 512 && tid < 256) s_sum[tid] += s_sum[tid + 256];
	__syncthreads();
	if (iBlockSize >= 256 && tid < 128) s_sum[tid] += s_sum[tid + 128];
	__syncthreads();
	if (iBlockSize >= 128 && tid < 64) s_sum[tid] += s_sum[tid + 64];
	__syncthreads();

	if (tid < 32)
	{
		volatile int *vsmem = s_sum;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}
	if (tid == 0) output[blockIdx.x] = s_sum[0];
}

/*Labels*/
__global__ void add_labels(int *d_labels, int lbl, int index) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	d_labels[idx + index] = lbl;
}

void knn_sort(int K, int &index, int lbl)
{
	checkCudaErrors(cudaMemcpy(d_norm_array, norm_array, ((sizeof(int))*WSIZE), cudaMemcpyHostToDevice));
	#if(SORTING == RADIX)
	RadixSort << <1, WSIZE >> >(d_norm_array, WSIZE);
	#elif(SORTING == INSERTION)
	InsertionSort << <1, WSIZE >> >(d_norm_array, WSIZE);
	#endif
	cudaMemcpy((d_k_norms + index), d_norm_array, (sizeof(int)*K), cudaMemcpyDeviceToDevice);
	add_labels << <1, K >> > (d_labels, lbl, index);
}

int perform_classification(int K, int num_classes) {
	int *cpu_labels = (int*)calloc(num_classes*K, sizeof(int));
	RadixSortLabels << <1, num_classes*K >> >(d_k_norms, d_labels, num_classes*K);

	checkCudaErrors(cudaMemcpy(cpu_labels, d_labels, ((sizeof(int)) * num_classes*K), cudaMemcpyDeviceToHost));
	int *count = (int*)calloc(num_classes, sizeof(int));
	for (int i = 0; i < K; i++) {
		count[cpu_labels[i]] += 1;
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

void knn_cuda(int *train_image, int dist_index)
{
	checkCudaErrors(cudaMemcpy(d_train, train_image, ((sizeof(int))*DIM * DIM), cudaMemcpyHostToDevice));
	dim3 block(THREADS, 1);
	dim3 grid((DIM * DIM + block.x - 1) / block.x, 1);
	auto start = std::chrono::high_resolution_clock::now();
	#if(METRIC == L1)
		knn_distance_L1 << < grid, block>> >(d_train, d_test, pixelnorm, DIM, DIM);
	#elif(METRIC == L2)
		knn_distance_L2 << < grid, block >> >(d_train, d_test, pixelnorm, DIM, DIM);
	#endif

	block.x = SUM_BLK_SIZE;
	grid.x = ((DIM * DIM) + block.x - 1) / block.x;
	WarpSumUnroll8<SUM_BLK_SIZE> << <grid.x / 8, block >> >(pixelnorm, d_odata, DIM * DIM);
	checkCudaErrors(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	int gpu_sum = 0;
	for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
	
	#if(METRIC == L1)
		norm_array[dist_index] = gpu_sum;
	#elif(METRIC == L2)
		norm_array[dist_index] = sqrt(gpu_sum);
	#endif
}

void cuda_deallocation() {
	cudaFree(d_train);
	cudaFree(d_test);
	cudaFree(pixelnorm);
	cudaFree(sum);
	cudaFree(norm_array);
	cudaFree(d_k_norms);
	cudaFree(d_labels);
}