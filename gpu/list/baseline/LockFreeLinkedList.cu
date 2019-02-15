/*

Copyright 2012-2013 Indian Institute of Technology Kanpur. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions, and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY INDIAN INSTITUTE OF TECHNOLOGY KANPUR ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INDIAN INSTITUTE OF TECHNOLOGY KANPUR OR
THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied, of Indian Institute of Technology Kanpur.

*/

/**********************************************************************************

 Lock-free linked list for CUDA; tested for CUDA 4.2 on 32-bit Ubuntu 10.10 and 64-bit Ubuntu 12.04.
 Developed at IIT Kanpur.

 Inputs: Percentage of add and delete operations (e.g., 30 50 for 30% add and 50% delete)
 Output: Prints the total time (in milliseconds) to execute the the sequence of operations

 Compilation flags: -O3 -arch sm_20 -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ -DNUM_ITEMS=num_ops -DFACTOR=num_ops_per_thread -DKEYS=num_keys

 NUM_ITEMS is the total number of operations (mix of add, delete, search) to execute.

 FACTOR is the number of operations per thread.

 KEYS is the number of integer keys assumed in the range [10, 9+KEYS].
 The paper cited below states that the key range is [0, KEYS-1]. However, we have shifted the range by +10 so that
 the head sentinel key (the minimum key) can be chosen as zero. Any positive shift other than +10 would also work.

 The include path ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ is needed for cutil.h.

 Related work:

 Prabhakar Misra and Mainak Chaudhuri. Performance Evaluation of Concurrent Lock-free Data Structures
 on GPUs. In Proceedings of the 18th IEEE International Conference on Parallel and Distributed Systems,
 December 2012.

***************************************************************************************/

#include<cuda_runtime.h>
#include<cuda.h>
#include<stdio.h>
#include "include/linkedlist.hpp"
#include "../include/cuda_intrinsics.h"

// The main kernel

__global__ void kernel(LL* items, LL* op, LL* result, Node** n)
{
  // The array items holds the sequence of keys
  // The array op holds the sequence of operations
  // The array result, at the end, will hold the outcome of the operations
  // n points to an array of pre-allocated free linked list nodes

  nodes=n;
  int tid;
  int i;
  for(i=0;i<FACTOR;i++){    		// FACTOR is the number of operations per thread
    tid=i*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=NUM_ITEMS) return;

    // Grab the operation and the associated key and execute
    LL itm=items[tid];
    if(op[tid]==ADD){
      result[tid]=list->Add(itm);
    }
    if(op[tid]==DELETE){
      result[tid]=list->Delete(itm);
    }
    if(op[tid]==SEARCH){
      result[tid]=list->Search(itm);
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) {
     printf("Need two arguments: percent add ops and percent delete ops (e.g., 30 50 for 30%% add and 50%% delete).\nAborting...\n");
     exit(1);
  }

  int adds=atoi(argv[1]);
  int deletes=atoi(argv[2]);

   if (adds+deletes > 100) {
     printf("Sum of add and delete precentages exceeds 100.\nAborting...\n");
     exit(1);
  }

  // Allocate linked list

  LinkedList* list=new LinkedList();
  LinkedList* Clist;
  int i;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&Clist, sizeof(LinkedList)));

	CHECK_CUDA_ERROR(cudaMemcpy(Clist, list, sizeof(LinkedList), cudaMemcpyHostToDevice));

  // Initialize the device memory
  init<<<1, 32>>>(Clist);

  LL op[NUM_ITEMS];		// Array of operations
  LL items[NUM_ITEMS];		// Array of keys associated with operations
  LL result[NUM_ITEMS];		// Array of outcomes

  srand(0);

  // NUM_ITEMS is the total number of operations to execute
  for(i=0;i<NUM_ITEMS;i++){
    items[i]=10+rand()%KEYS;	// Keys
  }

  // Populate the op sequence
  for(i=0;i<(NUM_ITEMS*adds)/100;i++){
    op[i]=ADD;
  }
  for(;i<(NUM_ITEMS*(adds+deletes))/100;i++){
    op[i]=DELETE;
  }
  for(;i<NUM_ITEMS;i++){
    op[i]=SEARCH;
  }

  adds=(NUM_ITEMS*adds)/100;

  // Allocate device memory

  LL* Citems;
  LL* Cop;
  LL* Cresult;
	
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS));
	CHECK_CUDA_ERROR(cudaMemcpy(Citems,items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));

  Node* pointers[adds];
  Node** Cpointers;

  // Allocate the pool of free nodes

	for(i=0;i<adds;i++){
		CHECK_CUDA_ERROR(cudaMalloc((void**)&pointers[i],sizeof(Node)));
  }
  
	CHECK_CUDA_ERROR(cudaMalloc((void**)&Cpointers, sizeof(Node*)*adds));
	CHECK_CUDA_ERROR(cudaMemcpy(Cpointers, pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice));

  // Calculate the number of thread blocks
  // NUM_ITEMS = total number of operations to execute
  // NUM_THREADS = number of threads per block
  // FACTOR = number of operations per thread

  int blocks=(NUM_ITEMS%(NUM_THREADS*FACTOR)==0)?NUM_ITEMS/(NUM_THREADS*FACTOR):(NUM_ITEMS/(NUM_THREADS*FACTOR))+1;

  // Launch main kernel

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  kernel<<<blocks, NUM_THREADS>>>(Citems, Cop, Cresult, Cpointers);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Print kernel execution time in milliseconds

  printf("%lf\n",time);

  // Check for errors

  cudaError_t error= cudaGetLastError();
  if(cudaSuccess!=error){
    printf("error:CUDA ERROR (%d) {%s}\n",error,cudaGetErrorString(error));
    exit(-1);
  }

  // Move results back to host memory
	CHECK_CUDA_ERROR(cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost));

  return 0;
}
