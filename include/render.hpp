#ifndef RENDERH
#define RENDERH

#include <stdint.h>

#include "hip-commons.h"
#include "screen.hpp"

constexpr static double TWOPI = M_PI * 2.0;

struct IPoint { int32_t x, y; };
struct FPoint {   float x, y; };
struct DPoint {  double x, y; };

struct ITri { IPoint a, b, c; };
struct FTri { FPoint a, b, c; };
struct DTri { DPoint a, b, c; };

constexpr static uint32_t colorBackground = (Color{  20, 20, 20,255 }).col();
constexpr static uint32_t colorPlayer     = (Color{ 240,  0, 70,255 }).col();
constexpr static uint32_t colorVisible    = (Color{  80, 80, 85,255 }).col();
constexpr static uint32_t colorObstacle   = (Color{  10, 10, 11,255 }).col();


__global__ void render(
    uint32_t* d_buf, const uint32_t w, const uint32_t h,
    FTri* d_tris, const uint32_t ntris,
    const float actorx, const float actory, const float actorr);


#endif // RENDERH
