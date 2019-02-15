//
// Created by Benjamin Trapani on 12/22/17.
//

#include "LockFreeQueue.hpp"
#include <array>
#include <iostream>

using namespace LockFreeQueueGPU;

typedef unsigned long long int SumElement_t;
typedef LockFreeQueue<SumElement_t, 0> Queue_t;

__global__ void reduceSum(SumElement_t *valuesToSum,
                          SumElement_t *result,
                          Queue_t** ppSharedQueue) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const SumElement_t valueHere = valuesToSum[tid];

    __shared__ int sharedListSize[1];
    __shared__ unsigned int totalAdditions[1];

    if (tid == 0) {
        sharedListSize[0] = blockDim.x;
        totalAdditions[0] = 0;
        *ppSharedQueue = new Queue_t();
    }
    __syncthreads();

    Queue_t* sharedQueue = *ppSharedQueue;

    sharedQueue->enqueue(valueHere);

    SumElement_t val1;
    SumElement_t val2;

    while(sharedListSize[0] > 1) {
        // When there are less elements in the queue than number of threads,
        // it becomes unlikely that a single thread captures both elements.
        // Subtract 1 from it to guarantee that at least one thread processes two elements.
        if (tid < sharedListSize[0] - 1) {
            while (true) {
                const bool popped1 = sharedQueue->dequeue(val1);
                const bool popped2 = sharedQueue->dequeue(val2);

                if (popped1 && popped2) {
                    const SumElement_t tempResult = val1 + val2;
                    atomicAdd(totalAdditions, 1);
                    sharedQueue->enqueue(tempResult);
                    atomicSub(sharedListSize, 1);
                } else {
                    if (popped1) {
                        sharedQueue->enqueue(val1);
                    }
                    if (popped2) {
                        sharedQueue->enqueue(val2);
                    }
                    break;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0){
        while(!sharedQueue->dequeue(result[0])){}
        result[1] = totalAdditions[0];
        delete *ppSharedQueue;
    }
}

template<size_t arraySize>
std::array<SumElement_t, arraySize> generateValuesToAdd(){
    std::array<SumElement_t, arraySize> result;
    for(size_t i = 0; i < arraySize; i++){
        result[i] = i + 1;
    }
    return result;
}

template<size_t numElements>
void runReduceSumTest(){
    const std::array<SumElement_t, numElements> generatedValues = generateValuesToAdd<numElements>();
    constexpr size_t expectedSum = (numElements + 1) * (numElements / 2);

    SumElement_t* valuesToSumOnDevice;
    if (cudaMalloc((void**) &valuesToSumOnDevice, sizeof(SumElement_t) * numElements) != 0){
        std::cerr << "Failed to allocate memory for values to sum" << std::endl;
    };
    if (cudaMemcpy(valuesToSumOnDevice, &generatedValues[0], sizeof(SumElement_t) * numElements, cudaMemcpyHostToDevice) != 0){
        std::cerr << "Failed to copy generated values to sum to device" << std::endl;
    }

    SumElement_t* sumResult;
    if (cudaMalloc((void**) &sumResult, sizeof(SumElement_t) * 2) != 0) {
        std::cerr << "Failed to allocate device memory for sum result" << std::endl;
    }
    if (cudaMemset(sumResult, 0, sizeof(SumElement_t) * 2) != 0){
        std::cerr << "Failed to zero device result memory" << std::endl;
    }

    Queue_t **ppQueue;
    if (cudaMalloc((void**) &ppQueue, sizeof(Queue_t*)) != 0) {
        std::cerr << "Failed to allocate pointer to queue" << std::endl;
    }

    reduceSum<<<1, numElements>>>(valuesToSumOnDevice, sumResult, ppQueue);
    cudaDeviceSynchronize();

    SumElement_t resultOnHost[2];
    cudaMemcpy(&resultOnHost, sumResult, sizeof(SumElement_t) * 2, cudaMemcpyDeviceToHost);
    std::cout << "Total additions: " << resultOnHost[1] << std::endl;
    std::cout << "Sum result: " << resultOnHost[0] << std::endl;
    std::cout << "Expected result: " << expectedSum << std::endl;
}

int main(int argc, char** argv){
    runReduceSumTest<128>();
    return 0;
}
