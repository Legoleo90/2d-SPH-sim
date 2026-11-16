#pragma once

__global__
void InitParticles(Particle* particles, curandState *states, int width, int height, const int particleCount);
__global__
void InitHashIndex(Index* indexBorders, const int particleCount);
__global__ 
void UpdatePosition(Particle* particles, int width, int height, const int particleCount, float dTime);
__global__
void GetDensityPressure(Particle* particles, const int particleCount, Index* hashBorders, unsigned int* indices, const int cellSize, const int width, const int height);
__global__
void GetPressure(Particle* particles, const int particleCount);
__global__
void UpdateDirection(Particle* particles, float dTime, const int cellSize, const int width, const int height, const int particleCount, Index* hashBorders, unsigned int* indices);
__global__
void CalculateHash(Particle* particles, const int width, const int height, unsigned int* hash, const int cellSize, const int particleCount);
__global__
void InitIndices(unsigned int* indices, int particleCount);
__host__
void SortHash(unsigned int* hash, unsigned int* indices, const int particleCount);
__global__
void GetHashBorders(unsigned int* sortedHash, const int particleCount, Index* indexBorders, const int hashCount);
__global__
void placeParticles(unsigned char* image, Particle* particles, const int width, const int height, const int channels, const int particleCount);
