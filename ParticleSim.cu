#include "Particle.cu"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <climits>

template<typename T>
__device__
inline T Clamp(T t, T a, T b) {
    return min(max(t, a), b);
}

__device__ __forceinline__
unsigned int GetHash(Vector2 position, int cellSize, int width, const int height, Vector2 Offset) {
    const int cellsPerRow = (width + cellSize - 1) / cellSize;
    const int cellsPerColumn = (height + cellSize - 1) / cellSize;

    int cellX = floorf(position.x) / cellSize + (int)Offset.x;
    int cellY = floorf(position.y) / cellSize + (int)Offset.y;

    if (cellX < 0 || cellX >= cellsPerRow ||
        cellY < 0 || cellY >= cellsPerColumn){
        return UINT_MAX;
    }
    
    return cellX + cellsPerRow * cellY;
}

__device__ __forceinline__
float Distance(Vector2 a, Vector2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

__global__ 
void InitParticles(Particle* particles, curandState *states, int width, int height, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    const float wallRepulsionDis = 0.1;
    curand_init(0, id, 0, &states[id]);
    float x = (curand_uniform(&states[id]) - 0.5) * 100 + width / 2;
    float y = curand_uniform(&states[id]) * (height - 2 * wallRepulsionDis) + wallRepulsionDis;
    const float startingVol = 5;
    particles[id] = Particle(x, y, startingVol);
}

__global__
void InitHashIndex(Index* indexBorders, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    indexBorders[id] = Index();
}

__global__
void InitIndices(unsigned int* indices, int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    indices[id] = id;
}

__global__ 
void UpdatePosition(Particle* particles, int width, int height, const int particleCount, float dTime){
    const float eps = 16;
    const float boundDamping = -0.5f;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    particles[id].velocity = particles[id].velocity + particles[id].force * dTime;
    particles[id].position = particles[id].position + particles[id].velocity * dTime;

    if (particles[id].position.x - eps < 0.f){
        particles[id].velocity.x *= boundDamping;
        particles[id].position.x = eps;
    }
    if (particles[id].position.x + eps > static_cast<float>(width)){
        particles[id].velocity.x *= boundDamping;
        particles[id].position.x = width - eps;
    }
    if (particles[id].position.y - eps < 0.f){
        particles[id].velocity.y *= boundDamping;
        particles[id].position.y = eps;
    }
    if (particles[id].position.y + eps > static_cast<float>(height)){
        particles[id].velocity.y *= boundDamping;
        particles[id].position.y = height - eps;
    }
}

__host__
void SortHash(unsigned int* hash, unsigned int* indices, const int particleCount){
    thrust::sort_by_key(
        thrust::device,
        hash,
        hash + particleCount,
        indices
    );
}
__global__
void GetDensityPressure(Particle* particles, const int particleCount, Index* hashBorders, unsigned int* indices, const int cellSize, const int width, const int height){
    const float smoothingLength = 16;
    const float mass = 25.f;
    const float gasConst = 2000.f;
    const float restDens = 300.f;
    const float pi = 3.14159;
    const float poly6 = 4.f / (pi * __powf(smoothingLength, 8.f)); 
    const float hsq = smoothingLength * smoothingLength;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    int neighbours = 0;
    float density = 0;
    for (int col = -1; col <= 1; col++){
        for (int row = -1; row <= 1; row++){
            unsigned int hash = GetHash(
                particles[id].position,
                cellSize, 
                width,
                height,
                Vector2(col, row)
            );
            if (hash == UINT_MAX)
                continue;
            for (int i = hashBorders[hash].start; i < hashBorders[hash].end; i++){
                unsigned int index = indices[i];
                float dis = Distance(particles[id].position, particles[index].position);
                float r2 = dis*dis;
                if (r2 < hsq){
                    neighbours++;
                    density += mass * poly6 * __powf(hsq - r2, 3.f);
                }
            }
        }
    }
    particles[id].density = density;
    particles[id].pressure = gasConst * (density - restDens);
    if (id == 0)
        printf("%f\n", density);
}

__global__
void UpdateDirection(Particle* particles, float dTime, const int cellSize, const int width, const int height, const int particleCount, Index* hashBorders, unsigned int* indices){    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;

    const float pi = 3.14159265f;
    const float smoothingLength = 16;
    const float mass = 2.5f;
    const float viscConst = 200.f;
    const float viscLap  = 40.f / (pi * __powf(smoothingLength, 5.f));
    const float spikyGrad = -10.f / (pi * __powf(smoothingLength, 5.f));

    Vector2 press = Vector2();
    Vector2 visc = Vector2();
    for (int col = -1; col <= 1; col++){
        for (int row = -1; row <= 1; row++){
            unsigned int hashVal = GetHash(particles[id].position, cellSize, width, height, Vector2(col, row));
            if (hashVal == UINT_MAX)
                continue;
            for (int i = hashBorders[hashVal].start; i < hashBorders[hashVal].end; i++){
                unsigned int index = indices[i];
                if (index == id)
                    continue;
                Vector2 rij = particles[index].position - particles[id].position;
                float r = Distance(particles[id].position, particles[index].position);
                if (r < smoothingLength){
                    press = press + rij.Normalized() * -mass * (particles[id].pressure + particles[index].pressure) / (2.f * particles[index].density) * spikyGrad * __powf(smoothingLength - r, 3.f);
                    visc = visc + (particles[index].velocity - particles[id].velocity) * (viscConst * mass / particles[index].density * viscLap * (smoothingLength - r));
                }
            }
        }
    }
    particles[id].force = (press + visc) / particles[id].density - Vector2(0.f, 9.82f);
}

__global__
void CalculateHash(Particle* particles, const int width, const int height, unsigned int* hash, const int cellSize, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    unsigned int hashVal = GetHash(particles[id].position, cellSize, width, height, Vector2());
    hash[id] = hashVal;
}

__global__
void GetHashBorders(unsigned int* sortedHash, const int particleCount, Index* indexBorders, const int hashCount)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;

    unsigned int currentHash = sortedHash[id];
    if (id == 0 || sortedHash[id - 1] != currentHash)
        indexBorders[currentHash].start = id;
    if (id == particleCount - 1 ||
        sortedHash[id + 1] != currentHash)
        indexBorders[currentHash].end = id + 1;
}

__global__
void placeParticles(unsigned char* image, Particle* particles, const int width, const int height, const int channels, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    const int radius = 4;
    if (id >= particleCount)
        return;
    int x = (int)particles[id].position.x;
    int y = (int)particles[id].position.y;
    for (int col = -radius; col <= radius; col++){
        for (int row = -radius; row <= radius; row++){
            if (col * col + row * row > radius * radius)
                continue;
            unsigned char* pixel = image + (Clamp(y + row, 0, height - 1) * width + Clamp(x + col, 0, width-1)) * channels;
            Color col = Color(80, 80, 255);
            pixel[0] = col.r;
            pixel[1] = col.g;
            pixel[2] = col.b;
        }
    }
}