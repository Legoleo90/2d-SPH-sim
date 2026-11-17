#include "Particle.cu"
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <climits>

__device__ const float h = 2.f;
__device__ const float mass = 2.5f;
__device__ const float gasConst = 2000.f;
__device__ const float restDens = 300.f;
__device__ const float pi = 3.14159f;
__device__ const float viscConst = 200.f;

static class Math{
public:
    template<typename T>
    __device__ __forceinline__
    static T Clamp(T t, T a, T b) {
        return min(max(t, a), b);
    }
    __device__ __forceinline__
    static float Distance(Vector2 a, Vector2 b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return sqrtf(dx * dx + dy * dy);
    }
};

static class Hash{
public:
    __device__ 
    static unsigned int GetHash(Vector2 position, int cellSize, int width, const int height, Vector2 Offset) {
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
};

//HASHING//
__global__ 
void InitParticles(Particle* particles, curandState *states, int width, int height, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    const int verticalParticles = height / h;
    const int horizontalParticles = (particleCount + verticalParticles - 1) / verticalParticles;
    curand_init(0, id, 0, &states[id]);
    const float xPos = (id % horizontalParticles) * h + width / 2 - (horizontalParticles / 2) * h + curand_uniform(&states[id]);
    const float yPos = (id / horizontalParticles) * h;
    particles[id] = Particle(xPos, yPos);
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
void CalculateHash(Particle* particles, const int width, const int height, unsigned int* hash, const int cellSize, const int particleCount){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    unsigned int hashVal = Hash::GetHash(particles[id].position, cellSize, width, height, Vector2());
    hash[id] = hashVal;
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

//FORCE CALUCULATIONS//
__global__
void GetDensityPressure(Particle* particles, const int particleCount, Index* hashBorders, unsigned int* indices, const int cellSize, const int width, const int height){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    const float poly6 = 4.f / (pi * __powf(h, 8.f)); 
    const float hsq = h * h;
    int neighbours = 0;
    float density = 0;
    for (int col = -1; col <= 1; col++){
        for (int row = -1; row <= 1; row++){
            unsigned int hash = Hash::GetHash(
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
                float dis = Math::Distance(particles[id].position, particles[index].position);
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
}
__global__
void UpdateDirection(Particle* particles, float dTime, const int cellSize, const int width, const int height, const int particleCount, Index* hashBorders, unsigned int* indices){    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;

    const float viscLap  = 40.f / (pi * __powf(h, 5.f));
    const float spikyGrad = -10.f / (pi * __powf(h, 5.f));

    Vector2 press = Vector2();
    Vector2 visc = Vector2();
    for (int col = -1; col <= 1; col++){
        for (int row = -1; row <= 1; row++){
            unsigned int hashVal = Hash::GetHash(particles[id].position, cellSize, width, height, Vector2(col, row));
            if (hashVal == UINT_MAX)
                continue;
            for (int i = hashBorders[hashVal].start; i < hashBorders[hashVal].end; i++){
                unsigned int index = indices[i];
                if (index == id)
                    continue;
                Vector2 rij = particles[index].position - particles[id].position;
                float r = Math::Distance(particles[id].position, particles[index].position);
                if (r < h){
                    press = press + rij.Normalized() * -mass * (particles[id].pressure + particles[index].pressure) / (2.f * particles[index].density) * spikyGrad * __powf(h - r, 3.f);
                    visc = visc + (particles[index].velocity - particles[id].velocity) * (viscConst * mass / particles[index].density * viscLap * (h - r));
                }
            }
        }
    }
    Vector2 gravForce = Vector2(0, -9.82f) * mass / particles[id].density;
    particles[id].force = (press + visc) / particles[id].density + gravForce;
}

//POSITION UPDATING
__global__ 
void UpdatePosition(Particle* particles, int width, int height, const int particleCount, float dTime){
    const float boundDamping = -0.5f;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= particleCount)
        return;
    particles[id].velocity = particles[id].velocity + particles[id].force * dTime;
    particles[id].position = particles[id].position + particles[id].velocity * dTime;

    if (particles[id].position.x - h < 0.f){
        particles[id].velocity.x *= boundDamping;
        particles[id].position.x = h;
    }
    if (particles[id].position.x + h > static_cast<float>(width)){
        particles[id].velocity.x *= boundDamping;
        particles[id].position.x = width - h;
    }
    if (particles[id].position.y - h < 0.f){
        particles[id].velocity.y *= boundDamping;
        particles[id].position.y = h;
    }
    if (particles[id].position.y + h > static_cast<float>(height)){
        particles[id].velocity.y *= boundDamping;
        particles[id].position.y = height - h;
    }
}

//IMAGE RENDERING//
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
            unsigned char* pixel = image + (Math::Clamp(y + row, 0, height - 1) * width + Math::Clamp(x + col, 0, width-1)) * channels;
            Color col = Color(80, 80, 255);
            pixel[0] = col.r;
            pixel[1] = col.g;
            pixel[2] = col.b;
        }
    }
}