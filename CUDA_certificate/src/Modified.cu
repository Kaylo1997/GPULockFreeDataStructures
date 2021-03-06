#include <stdio.h>
#include <stdlib.h>

#define block_size   1024

__global__ void calculation(    int *a, 
                                int *b, 
                                int *c, 
                                int constant, 
                                int vector_size ) {
        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
    // this thread handles the data at its thread id		
	__shared__ float myblock1[block_size+2];

	myblock1[tid] = a[tid];
	

	__syncthreads();

        if (tid < vector_size){
        
                // Read in inputs
                int prev_a = myblock1[tid>0?tid-1:(vector_size-1)];
                int curr_a = myblock1[tid];
                int post_a = myblock1[tid<(vector_size-1)?tid+1:0];
                
                int curr_b = b[tid];
                
                // Do computation
                int output_c = (prev_a-post_a)*curr_b + curr_a*constant;
                
                // Write result
                c[tid] = output_c;               
        }

	a[tid] = myblock1[tid];
}

int main( int argc, char* argv[] ) { 

        // Parse Input arguments
        
        // Check the number of arguments (we only receive command + vector size)
        if (argc != 2) {
                // Tell the user how to run the program
                printf ("Usage: %s vector_size\n", argv[0]);
                // "Usage messages" are a conventional way of telling the user
                // how to run a program if they enter the command incorrectly.
                return 1;
        }
        // Set GPU Variables based on input arguments
        int vector_size = atoi(argv[1]);
        int grid_size   = ((vector_size-1)/block_size) + 1;
                
        // Set device that we will use for our cuda code
        // It will be either 0 or 1
        cudaSetDevice(0);
        
	// Time Variables
	cudaEvent_t start_cpu, start_gpu;
        cudaEvent_t  stop_cpu,  stop_gpu;
	
        cudaEventCreate (&start_cpu);
	cudaEventCreate (&start_gpu);
        
	cudaEventCreate (&stop_cpu);
	cudaEventCreate (&stop_gpu);
	
	float time;
        
        // Input Arrays and variables
        int *a          = new int [vector_size]; 
        int *b          = new int [vector_size]; 
        int *c_cpu      = new  int [vector_size]; 
        int *c_gpu      = new  int [vector_size];
        int constant    = 4;

        // Pointers in GPU memory
        int *dev_a;
        int *dev_b;
        int  *dev_c;

        // fill the arrays 'a' and 'b' on the CPU
	printf("Filling up input arrays with random values between 1 and 10.\n");
        for (int i = 0; i < vector_size; i++) {
                a[i] = rand()%10;
                b[i] = rand()%10;
        }

        //
        // CPU Calculation
        //////////////////
        
	printf("Running sequential job.\n");
	cudaEventRecord(start_cpu,0);
        
        // Calculate C in the CPU
        for (int i = 0; i < vector_size; i++) {
                // Read in inputs
                int prev_a = a[i>0?i-1:(vector_size-1)];
                int curr_a = a[i];
                int post_a = a[i<(vector_size-1)?i+1:0];
                
                int curr_b = b[i];
                
                // Do computation
                int output_c = (prev_a-post_a)*curr_b + curr_a*constant;
                
                // Write result
                c_cpu[i] = output_c;
        }
        
	cudaEventRecord(stop_cpu,0);
	cudaEventSynchronize(stop_cpu);
        
	cudaEventElapsedTime(&time, start_cpu, stop_cpu);
	printf("\tSequential Job Time: %.2f ms\n", time);
      
        //
        // GPU Calculation
        //////////////////
        
        printf("Running parallel job.\n");
        
	cudaEventRecord(start_gpu,0);
        
        // allocate the memory on the GPU
        cudaMalloc( (void**)&dev_a,       vector_size * sizeof(int) );
        cudaMalloc( (void**)&dev_b,       vector_size * sizeof(int) );
        cudaMalloc( (void**)&dev_c,       vector_size * sizeof(int) );

        // set arrays to 0
        cudaMemset(dev_a,         0, vector_size * sizeof(int));
        cudaMemset(dev_b,         0, vector_size * sizeof(int));
        cudaMemset(dev_c,         0, vector_size * sizeof(int));
        
        // copy the arrays 'a' and 'b' to the GPU
        cudaMemcpy( dev_a, a, vector_size * sizeof(int),
                              cudaMemcpyHostToDevice );
        cudaMemcpy( dev_b, b, vector_size * sizeof(int),
                              cudaMemcpyHostToDevice );
        // run kernel
        calculation<<<grid_size,block_size>>>(  dev_a, 
                                                dev_b, 
                                                dev_c, 
                                                constant,
                                                vector_size );
                                                        
        // copy the array 'c' back from the GPU to the CPU
        cudaMemcpy( c_gpu, dev_c, vector_size * sizeof(int),
                              cudaMemcpyDeviceToHost );

	cudaEventRecord(stop_gpu,0);
	cudaEventSynchronize(stop_gpu);

	cudaEventElapsedTime(&time, start_gpu, stop_gpu);
	printf("\tParallel Job Time: %.2f ms\n", time);

        // compare the results
        int error = 0;
        for (int i = 0; i < vector_size; i++) {
                if (c_cpu[i] != c_gpu[i]){
                        error = 1;
                        printf( "Error starting element %d, %d != %d\n", i, c_gpu[i], c_cpu[i] );    
                }
		if (error) break; 
        }
        
        if (error == 0){
                printf ("Correct result. No errors were found.\n");
        }

        // free the memory allocated on the GPU
        cudaFree( dev_a );
        cudaFree( dev_b );
        cudaFree( dev_c );
        
        // free cuda events
        cudaEventDestroy (start_cpu);
	cudaEventDestroy (start_gpu);
        
	cudaEventDestroy (stop_cpu);
	cudaEventDestroy (stop_gpu);
        
        // free CPU memory        
	free(a);
	free(b);
	free(c_cpu);
	free(c_gpu);
	

       return 0;
}
