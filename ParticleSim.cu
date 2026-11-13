#include "Particle.cu"
#include <curand_kernel.h>
#include <thrust/sort.h>

template<typename T>
__device__
inline T Clamp(T t, T a, T b) {
    return min(max(t, a), b);
}

__device__ __forceinline__
unsigned int GetHash(int x, int y, int cellSize, int width) {
    int cellX = x / cellSize;
    int cellY = y / cellSize;
    int cellsPerRow = width / cellSize;
    return cellX + cellsPerRow * cellY;
}

__device__ __forceinline__
float Distance(Vector2 a, Vector2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

__device__
Vector2 BorderForce(Vector2 pos, const int width, const int height){
    const float maxDis = 30;
    const float repulsiveForce = 0.5;

    Vector2 force = Vector2(0, 0);

    if (pos.x < maxDis)
        force.x += repulsiveForce * (maxDis - pos.x);
    if (width - pos.x < maxDis)
        force.x -= repulsiveForce * (maxDis - (width - pos.x));
    if (pos.y < maxDis)
        force.y += repulsiveForce * (maxDis - pos.y);
    if (height - pos.y < maxDis)
        force.y -= repulsiveForce * (maxDis - (height - pos.y));

    return force;
}

__global__ 
void InitParticles(Particle* particles, curandState *states, int width, int height, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    const float wallRepulsionDis = 0.1;
    curand_init(0, id, 0, &states[id]);
    float x = (curand_uniform(&states[id]) - 0.5) * 20 + width / 2;
    float y = curand_uniform(&states[id]) * (height - wallRepulsionDis) + wallRepulsionDis;
    const float startingVol = 500;
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
void UpdatePosition(Particle* particles, int width, int height, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    particles[id].position.Add(particles[id].direction);
    float newX = Clamp(particles[id].position.x, (float)0, (float)width - 1);
    float newY = Clamp(particles[id].position.y, (float)0, (float)height - 1);
    particles[id].position.x = newX;
    particles[id].position.y = newY;

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

__device__
float poly6Kernel(float r, float h){
    if (r < 0 || h < r)
        return 0;
    const float pi = 3.14159;
    return (4.0 / (pi*__powf(h, 8))) * __powf(h*h -r*r, 3);
}

__global__
void GetDensity(Particle* particles, const int particleCount, Index* hashBorders, unsigned int* indices, const int cellSize, const int width){
    const float smoothingLength = 100;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;

    float density = 0;
    for (int col = -1; col <= 1; col++){
        for (int row = -1; row <= 1; row++){
            unsigned int hash = GetHash(
                particles[id].position.x + col,
                particles[id].position.y + row,
                cellSize, 
                width
            );
            for (int i = hashBorders[hash].start; i < hashBorders[hash].end; i++){
                unsigned int index = indices[i];
                float mass = particles[index].volume * particles[index].density;
                float dis = Distance(particles[id].position, particles[index].position);
                float kernel = poly6Kernel(dis, smoothingLength);
                density += mass * kernel;
            }
        }
    }
    particles[id].density = density;
}

__global__
void GetPressure(Particle* particles, const int particleCount){
    const float refDensity = 1000;
    const float soundSpeed = 20;
    const float y = 7;

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
        
    float pressure = (((soundSpeed * soundSpeed * refDensity) / y) * (__powf(particles[id].density / refDensity, y) - 1));
    particles[id].pressure = pressure;
}

__device__ __forceinline__
Vector2 Poly6Gradient(Vector2 rij, float r, float h) {
    if (r <= 1e-6f || r >= h)
        return Vector2();
    const float pi = 3.14159265f;
    float diff = h * h - r * r;
    float factor = -24.0f / (pi * __powf(h, 8)) * diff * diff;
    return rij * factor;
}

__global__
void UpdateDirection(Particle* particles, float dTime, const int cellSize, const int width, const int height, const int particleCount, Index* hashBorders, unsigned int* indices){    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    const float smoothingLength = 100;
    Vector2 dir = Vector2();
    for (int col = -1; col <= 1; col++){
        for (int row = -1; row <= 1; row++){
            const int cellsPerRow = (width + cellSize - 1) / cellSize;
            const int cellsPerColumn = (height + cellSize - 1) / cellSize;

            int cellX = static_cast<int>(particles[id].position.x) / cellSize + col;
            int cellY = static_cast<int>(particles[id].position.y) / cellSize + row;

            if (cellX < 0 || cellX >= cellsPerRow ||
                cellY < 0 || cellY >= cellsPerColumn)
                continue;

            unsigned int hashVal = cellX + cellsPerRow * cellY;
            for (int i = hashBorders[hashVal].start; i < hashBorders[hashVal].end; i++){
                unsigned int index = indices[i];
                float dis = Distance(particles[index].position, particles[id].position);
                Vector2 kernelGradient = Poly6Gradient(particles[id].position-particles[index].position, dis, smoothingLength);
                Vector2 newDirection = kernelGradient *(particles[id].pressure / __powf(particles[id].density, 2) + particles[index].pressure / __powf(particles[index].density, 2)) * particles[index].volume * particles[index].density * -1.0f;
                dir = dir + newDirection;
            }
        }
    }
    const float speed = 1;
    Vector2 gravity = Vector2(0.0f, -9.8f);
    Vector2 acceleration = dir * speed + gravity + BorderForce(particles[id].position, width, height);
    Vector2 newDir = Vector2(
        particles[id].direction.x + acceleration.x * dTime,
        particles[id].direction.y + acceleration.y * dTime
    );

    particles[id].direction = newDir;
}

__global__
void CalculateHash(Particle* particles, int width, unsigned int* hash, const int cellSize, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    unsigned int hashVal = GetHash(particles[id].position.x, particles[id].position.y, cellSize, width);
    hash[id] = hashVal;
}

__global__
void GetHashBorders( unsigned int* sortedHash, const int particleCount, Index* indexBorders, const int hashCount)
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
    const int radius = 1;
    if (id > particleCount)
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