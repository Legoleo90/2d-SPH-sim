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
};
struct Particle{
public: 
    Vector2 position, direction;
    float volume, pressure, density;
    __device__
    Particle(float x, float y, float vol) : position(Vector2(x, y)), direction(Vector2()), volume(vol), pressure(0), density(1010){};
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