#pragma once

struct Vector2{
public:
    float x, y;
    __device__
    Vector2(float X, float Y) : x(X), y(Y) {}
    __device__
    Vector2() : x(0), y(0) {}
    __device__ 
    void Add(Vector2 term){
        x += term.x;
        y += term.y;
    }
    __device__
    inline void Normalize(){
        float len = sqrtf(x*x + y*y);
        x /= len;
        y /= len;
    }
    __device__
    inline Vector2 Normalized(){
        float len = sqrtf(x*x + y*y);
        return Vector2(x / len, y / len);
    }
    __device__
    Vector2 operator+(const Vector2& term) const {
        return {
            x + term.x,
            y + term.y
        };
    }
    __device__
    Vector2 operator-(const Vector2& term) const {
        return {
            x - term.x,
            y - term.y
        };
    }
    __device__
    Vector2 operator/(const float& factor) const {
        return {
            x / factor,
            y / factor
        };
    }
    __device__
    Vector2 operator*(const float& factor) const {
        return {
            x * factor,
            y * factor
        };
    }
    __device__
    Vector2 operator*(const Vector2 factor) const {
        return {
            x * factor.x,
            y * factor.y
        };
    }
};
struct Particle{
public: 
    Vector2 position, force, velocity;
    float pressure, density, mass;
    __device__
    Particle(float x, float y) : position(Vector2(x, y)), force(Vector2()), velocity(Vector2()), pressure(0), density(1010), mass(2.5f){};
};
struct Index{
public:
    int start, end;
    __device__
    Index() : start(0), end(0) {}
};
struct Color{
public:
    int r, g, b;
    __device__
    Color(int red, int green, int blue) : r(red), g(green), b(blue) {}
};