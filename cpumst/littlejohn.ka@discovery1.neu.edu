#include <stdio.h>
#include <stdlib.h>

__global__ void reduction(float *input, float *result, size_t n){

  extern __shared__ float sdata[];
  int i = tid + blockIdx.x*blockDim.x;
  int tid = threadIdx.x;

  //loads input number into shared memory
  float x = 0;
  if(i < n){
    x = input[i];
  }
  sdata[tid] = x;

  __syncthreads();

  //reduction for each block in shared mem, shift right for powers of 2
  for(int off = blockDim.x/2; off > 0; off >>= 1){
    if(tid < off){
      sdata[tx] = sdata[tx] + sdata[tx+off]; //sum numbers and store in array
    }
  }

  __syncthreads();

  //now once at thread 0, write back into result
  if(threadIdx.x == 0){
    resuls[blockIdx.x] = sdata[0];
  }
}

int main(){

  int *input, *dev_input, *result, *dev_result;
  int n = 1024;

  size_t array_size = n *sizeof(int);

  input = (int*) malloc(n*sizeof(int));
  result = (int*) malloc(n*sizeof(int));

  for(int i = 0; i < n; i++){
    input[i] = i;
    result[i] = 0;
  }

  cudaMalloc((void**)&dev_input, array_size);
  cudaMalloc((void**)&dev_resul, array_size);

  cudaMemcpy(dev_input, input, array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_result, result, array_size, cudaMemcpyHostToDevice);

  int threads = 1024;
  int blocks = n/threads;
  int shared = threads * sizeof(int);

  reduction<<<blocks,threads,shared>>>(dev_input,dev_result);

  cudaMemcpy(input, dev_input, array_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(dev_result, result, array_size, cudaMemcpyDeviceToHost);

  printf("\nSum: %d\n", result[0]);

  return 0;
}
