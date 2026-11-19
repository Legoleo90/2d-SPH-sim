#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include "Particle.cu"
#include "ParticleSim.cuh"
#include <cub/cub.cuh>

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
Particle gpuParticles = NULL;
unsigned int*  hashValues = nullptr;
unsigned int* sortedHashValues = nullptr;
unsigned int* indices = nullptr;
unsigned int* sortedIndices = nullptr;
Index* hashBorders = nullptr;
void* tempStorage = nullptr;
size_t tempStorageByte = 0;

const int particleCount = 100000;
const int cellSize = 8;
const float deltaTime = 0.00005f;
const int subframes = 64;
const int sortingFreq = 32;
int cellCount = 0;

__global__
void placeParticles(unsigned char* image, Particle* particles, int width, int channels, const int particleCount);
__global__
void GenerateCudaImage(unsigned char* image, int width, int height, int channels);
void Hash(const int width, const int height);
void SimulateForces(const int width, const int height);
void InitSim(const int width, const int height, const int channels);
void Render(const int width, const int height, const int channels);

extern "C" void ClearCudaData(){
    cudaFree(gpuImagePtr);
    cudaFree(hashValues);
    cudaFree(hashBorders);
    cudaFree(indices);
    cudaFree(tempStorage);
    cudaFree(sortedHashValues);
    cudaFree(sortedIndices);
    gpuParticles.Free();
}

extern "C" unsigned char* GenerateImage(int width, int height, int channels){    
    if (gpuImagePtr == nullptr)
        InitSim(width, height, channels);
    for (int i = 0; i < subframes; i++){
        if (i % sortingFreq == 0)
            Hash(width, height);
        SimulateForces(width, height);
    }
    Render(width, height, channels);
    return gpuImagePtr;
}
void InitSim(const int width, const int height, const int channels){
    dim3 particleBlock(256);
    dim3 particleGrid(particleCount / particleBlock.x + 1);

    size_t imageSize = width * height * channels * sizeof(unsigned char);
    cudaMalloc(&gpuImagePtr, imageSize);

    gpuParticles = Particle(particleCount);

    curandState* states;
    cudaMalloc(&states, sizeof(curandState) * particleCount);

    InitParticles<<<particleGrid, particleBlock>>>(
        gpuParticles,
        states,
        width,
        height,
        particleCount
    );

    dim3 hashBlock(256);
    dim3 hashGrid(cellCount / hashBlock.x + 1);

    size_t hashSize = sizeof(unsigned int) * particleCount;
    cudaMalloc(&hashValues, hashSize);

    int cellWidth = (width + cellSize - 1) / cellSize; 
    int cellHeight = (height + cellSize - 1) / cellSize; 
    cellCount = cellWidth * cellHeight;
    size_t hashBorderSize = sizeof(Index) * cellCount;
    cudaMalloc(&hashBorders, hashBorderSize);

    InitHashIndex<<<hashGrid, hashBlock>>>(
        hashBorders,
        cellCount
    );

    cudaMalloc(&indices, hashSize);
    InitIndices<<<particleGrid, particleBlock>>>(
        indices,
        particleCount
    );

    cudaMalloc(&sortedIndices, hashSize);
    cudaMalloc(&sortedHashValues, hashSize);
}

void Hash(const int width, const int height){
    dim3 particleBlock(256);
    dim3 particleGrid(particleCount / particleBlock.x + 1);
    CalculateHash<<<particleGrid, particleBlock>>>(
        gpuParticles,
        width,
        height,
        hashValues,
        cellSize,
        particleCount
    );
    cudaDeviceSynchronize();
    InitIndices<<<particleGrid, particleBlock>>>(
        indices,
        particleCount
    );
    cudaDeviceSynchronize();
    if (tempStorage == nullptr){
        cub::DeviceRadixSort::SortPairs(
            tempStorage,
            tempStorageByte,
            hashValues,
            sortedHashValues,
            indices,
            sortedIndices,
            particleCount
        );
        cudaMalloc(&tempStorage, tempStorageByte);
    }
    SortHash(
        hashValues,
        sortedHashValues,
        indices,
        sortedIndices,
        tempStorage,
        tempStorageByte,
        particleCount
    );
    cudaDeviceSynchronize();
    cudaMemset(hashBorders, 0, sizeof(Index) * cellCount);
    GetHashBorders<<<particleGrid, particleBlock>>>(
        sortedHashValues,
        particleCount,
        hashBorders,
        cellCount
    );
    cudaDeviceSynchronize();
}

void SimulateForces(const int width, const int height){
    dim3 particleBlock(256);
    dim3 particleGrid(particleCount / particleBlock.x + 1);
    GetDensityPressure<<<particleGrid, particleBlock>>>(
        gpuParticles,
        particleCount,
        hashBorders,
        sortedIndices,
        cellSize,
        width,
        height
    );
    cudaDeviceSynchronize();
    UpdateDirection<<<particleGrid, particleBlock>>>(
        gpuParticles,
        deltaTime,
        cellSize,
        width,
        height,
        particleCount,
        hashBorders,
        sortedIndices
    );
    cudaDeviceSynchronize();
    UpdatePosition<<<particleGrid, particleBlock>>>(
        gpuParticles,
        width,
        height,
        particleCount,
        deltaTime
    );
    cudaDeviceSynchronize();
}

void Render(const int width, const int height, const int channels){
    dim3 imBlock(16, 16);
    dim3 imGrid(width / imBlock.x + 1, height / imBlock.y + 1);

    dim3 particleBlock(256);
    dim3 particleGrid(particleCount / particleBlock.x + 1);

    GenerateCudaImage<<<imGrid, imBlock>>>(
        gpuImagePtr,
        width,
        height,
        channels
    );
    placeParticles<<<particleGrid, particleBlock>>>(
        gpuImagePtr,
        gpuParticles,
        width,
        height,
        channels,
        particleCount
    );
}