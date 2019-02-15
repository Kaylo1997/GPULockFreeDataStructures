#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define block_size   32
#define vector_size  100


__global__ void add( int *a, int *b, int *c ) {
        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;    // this thread handles the data at its thread
id

        if (tid < vector_size){
                c[tid] = a[tid] + b[tid];                   // add vectors together
        }
}

int main( void ) {

        // Set device that we will use for our cuda code
        // It will be either 0 or 1
        cudaSetDevice(0);

        // Time Variables
        cudaEvent_t start, stop;

        cudaEventCreate (&start);
        cudaEventCreate (&stop);

        // Input Arrays and variables
        int *a        = new int [vector_size];
        int *b        = new int [vector_size];
        int *c          = new int [vector_size];

        // Pointers in GPU memory
        int *dev_a;
        int *dev_b;
        int *dev_c;

        // fill the arrays 'a' and 'b' on the CPU
        for (int i = 0; i < vector_size; i++) {
                a[i] = rand()%10;
                b[i] = rand()%10;
        }

    /* allocate space for device copies of a, b, c */
            cudaMalloc( (void **) &dev_a, vector_size*sizeof(int) );
    cudaMalloc( (void **) &dev_b, vector_size*sizeof(int ));
    cudaMalloc( (void **) &dev_c, vector_size*sizeof(int ));

        cudaMemcpy ( dev_a,  a, vector_size*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy ( dev_b,  b, vector_size*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy ( dev_c,  c, vector_size*sizeof(int), cudaMemcpyHostToDevice);
      //////////////////
        add <<< 4, 32>>> (dev_a, dev_b, dev_c);

         cudaMemcpy ( a,  dev_a, vector_size*sizeof(int), cudaMemcpyDeviceToHost);
         cudaMemcpy ( b,  dev_b, vector_size*sizeof(int), cudaMemcpyDeviceToHost);
         cudaMemcpy ( c,  dev_c, vector_size*sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree ( dev_a);
        cudaFree ( dev_a);
        cudaFree ( dev_a);

        printf("Running sequential job.\n");

        cudaEventRecord(start,0);
}
