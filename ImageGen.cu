#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include "Particle.cu"
#include "ParticleSim.cuh"

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " \
                  << cudaGetErrorString(err) << "\n" \
                  << "File: " << __FILE__ << "\n" \
                  << "Line: " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

unsigned char* gpuImagePtr = nullptr;
Particle* gpuParticles = nullptr;
unsigned int*  hashValues = nullptr;
unsigned int* indices = nullptr;
Index* hashBorders = nullptr;

const int particleCount = 10000;
const int cellSize = 8;
int cellCount = 0;

__global__
void placeParticles(unsigned char* image, Particle* particles, int width, int channels, const int particleCount);
__global__
void GenerateCudaImage(unsigned char* image, int width, int height, int channels);

extern "C" void ClearCudaData(){
    cudaFree(gpuImagePtr);
    cudaFree(gpuParticles);
    cudaFree(hashValues);
    cudaFree(hashBorders);
    cudaFree(indices);
}

extern "C" unsigned char* GenerateImage(int width, int height, int channels, float deltaTime){    
    deltaTime = 0.0007f;
    int subframes = 64;
    
    dim3 imBlock(16, 16);
    dim3 imGrid(width / imBlock.x + 1, height / imBlock.y + 1);

    dim3 particleBlock(256);
    dim3 particleGrid(particleCount / particleBlock.x + 1);
    CUDA_CHECK(cudaGetLastError());
    if (gpuImagePtr == nullptr){
        std::cout<<"FIrst running"<<std::endl;
        size_t imageSize = width * height * channels * sizeof(unsigned char);
        cudaMalloc(&gpuImagePtr, imageSize);

        size_t particleSize = sizeof(Particle) * particleCount;
        cudaMalloc(&gpuParticles, particleSize);
        curandState* states;
        cudaMalloc(&states, sizeof(curandState) * particleCount);
        CUDA_CHECK(cudaGetLastError());
        InitParticles<<<particleGrid, particleBlock>>>(
            gpuParticles,
            states,
            width,
            height,
            particleCount
        );
        CUDA_CHECK(cudaGetLastError());

        size_t hashSize = sizeof(unsigned int) * particleCount;
        cudaMalloc(&hashValues, hashSize);
        CUDA_CHECK(cudaGetLastError());

        int cellWidth = (width + cellSize - 1) / cellSize; 
        int cellHeight = (height + cellSize - 1) / cellSize; 
        cellCount = cellWidth * cellHeight;
        size_t hashBorderSize = sizeof(Index) * cellCount;
        cudaMalloc(&hashBorders, hashBorderSize);

        dim3 hashBlock(256);
        dim3 hashGrid(cellCount / hashBlock.x + 1);

        InitHashIndex<<<hashGrid, hashBlock>>>(
            hashBorders,
            cellCount
        );
        CUDA_CHECK(cudaGetLastError());
        cudaMalloc(&indices, hashSize);
        InitIndices<<<particleGrid, particleBlock>>>(
            indices,
            particleCount
        );
        CUDA_CHECK(cudaGetLastError());
    }
    GenerateCudaImage<<<imGrid, imBlock>>>(
        gpuImagePtr,
        width,
        height,
        channels
    );
    for (int i = 0; i < subframes; i++){
        CalculateHash<<<particleGrid, particleBlock>>>(
            gpuParticles,
            width,
            height,
            hashValues,
            cellSize,
            particleCount
        );
        InitIndices<<<particleGrid, particleBlock>>>(
            indices,
            particleCount
        );
        SortHash(
            hashValues,
            indices,
            particleCount
        );
        cudaMemset(hashBorders, 0, sizeof(Index) * cellCount);
        GetHashBorders<<<particleGrid, particleBlock>>>(
            hashValues,
            particleCount,
            hashBorders,
            cellCount
        );
        GetDensityPressure<<<particleGrid, particleBlock>>>(
            gpuParticles,
            particleCount,
            hashBorders,
            indices,
            cellSize,
            width,
            height
        );
        UpdateDirection<<<particleGrid, particleBlock>>>(
            gpuParticles,
            deltaTime,
            cellSize,
            width,
            height,
            particleCount,
            hashBorders,
            indices
        );
        UpdatePosition<<<particleGrid, particleBlock>>>(
            gpuParticles,
            width,
            height,
            particleCount,
            deltaTime
        );
    }
    placeParticles<<<particleGrid, particleBlock>>>(
        gpuImagePtr,
        gpuParticles,
        width,
        height,
        channels,
        particleCount
    );
    return gpuImagePtr;
}

__global__
void GenerateCudaImage(unsigned char* image, int width, int height, int channels){
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    int globalId = idY * width + idX;


    if (idX >= width || idY >= height)
        return;
   
    unsigned char* pixel = image + (globalId) * channels;
    pixel[0] = 25;
    pixel[1] = 25;
    pixel[2] = 25;
    pixel[3] = 255;
}

