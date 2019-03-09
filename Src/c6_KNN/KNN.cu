#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include<cmath>

int* knn_dist;
int* ref_device;
int* test_device;
int* diff;
int* sum;
__device__ int dist = 0;


__global__ void knn_distance(int *ref, int *test, int *diff, int nx,  int ny)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;
	//printf("%d\t", test[ix]);
	//__shared__ float ref_shared[BLOCK_DIM];
	//__shared__ float test_shared[BLOCK_DIM];
	if (ix < nx)
	{
		for (int iy = 0; iy < ny; iy++)
		{
			int index = iy * nx + ix;
			diff[index] = (test[index] - ref[index]);
		}
	}
}


__global__ void warp_sum(int *diff, int *sum, int nx, int ny)
{
	int laneid = (threadIdx.x + blockDim.x*blockIdx.x);
	__shared__ int buf[2048];
	//buff[laneid] = sum[laneid];

	for (int j = 0; j < 2048; j++)
	{
		int temp = 0;
		int dif = diff[32*j + laneid%32];
		for (int i = 1; i <= 32; i *= 2)
		{
			temp = __shfl_up(dif, i, 32);
			if ((laneid%32) >= i)
				dif = dif + temp;
		}
		buf[j] = __shfl(dif, 31);
		//sum[j] = buf[j];
	}
	__syncthreads();
	if (laneid < 2048) {
		for (int j = 0; j < 64; j++)
		{
			int temp = 0;
			int dif = buf[32 * j + laneid % 32];
			for (int i = 1; i <= 32; i *= 2)
			{
				temp = __shfl_up(dif, i, 32);
				if ((laneid % 32) >= i)
					dif = dif + temp;
			}
			buf[j] = __shfl(dif, 31);
			//sum[j] = buf[j];
		}
	}
	__syncthreads();
	if (laneid < 64) {
		for (int j = 0; j < 2; j++)
		{
			int temp = 0;
			int dif = buf[32 * j + laneid % 32];
			for (int i = 1; i <= 32; i *= 2)
			{
				temp = __shfl_up(dif, i, 32);
				if ((laneid % 32) >= i)
					dif = dif + temp;
			}
			buf[j] = __shfl(dif, 31);
			sum[j] = buf[j];
		}
	}
	__syncthreads();
	dist = sum[0] + sum[1];
}

void knn_cuda(int * ref, int *test)
{
	cudaError_t err0 = cudaSetDevice(0);
	if (err0 != cudaSuccess)
	{
		printf("ERROR: Cannot set the chosen CUDA device\n");
	}

	int dim = 256;
	checkCudaErrors(cudaMalloc((void **)&ref_device, (sizeof(int) * dim * dim)));
	checkCudaErrors(cudaMalloc((void **)&test_device, (sizeof(int) * dim * dim)));
	checkCudaErrors(cudaMalloc((void **)&diff, (sizeof(int) * dim*dim)));
	checkCudaErrors(cudaMalloc((void **)&sum, (sizeof(int) * 2)));

	checkCudaErrors(cudaMemcpy(ref_device, ref, ((sizeof(int))*dim * dim), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(test_device, test, ((sizeof(int))*dim * dim), cudaMemcpyHostToDevice));

	dim3 block(128, 1, 1);
	dim3 grid((dim + block.x - 1) / block.x, 1, 1);

	knn_distance << < grid, block >> >(ref_device, test_device, diff, dim, dim);

	int * cpu_diff = (int*)malloc(sizeof(int)*dim*dim);
	cudaMemcpy(cpu_diff, diff, (sizeof(int)*dim * dim), cudaMemcpyDeviceToHost);


	warp_sum << <256, 256>> > (diff, sum, dim, dim);


	/*int * cpu_sum = (int*)malloc(sizeof(int)* 2048);
	cudaMemcpy(cpu_sum, sum, (sizeof(int)* 2048), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < 2; i++)
	{
		sum += cpu_sum[i];
	}
	printf("Sum Now:%d\n", sum);*/

	int d = 0;
	cudaMemcpyFromSymbol(&d, dist, sizeof(int));
	printf("%d\n", d);

	//checkCudaErrors(cudaFree(ref_device));
//	checkCudaErrors(cudaFree(test_device);

	//checkCudaErrors(cudaDeviceReset());

}

