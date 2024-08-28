#include "render.hpp"

#include <cfloat>


// https://stackoverflow.com/a/2049593
constexpr __device__ static inline float sign(
    const FPoint p1, const FPoint p2, const FPoint p3)
{
    return (float)(p1.x - p3.x) * (float)(p2.y - p3.y) -
           (float)(p2.x - p3.x) * (float)(p1.y - p3.y);
}
constexpr __device__ static inline bool withinTri(
    const FPoint p, const FTri t)
{
    const float d1 = sign(p, t.a, t.b);
    const float d2 = sign(p, t.b, t.c);
    const float d3 = sign(p, t.c, t.a);

    const bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    const bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}


__device__ static inline void pointRelatives(
    const FPoint source, const FPoint target,
    float& distance, float& angle)
{
    const float dx = target.x - source.x;
    const float dy = target.y - source.y;
    distance = hypotf(dx, dy);
    angle = atan2f(dy, dx);
}


// represents ax + by + c = 0
struct FStandardFormLine
{
    double a, b, c;
};
constexpr __device__ static inline FStandardFormLine eqTwoPoints(
    const FPoint a, const FPoint b)
{
    const double A    = b.y - a.y;
    const double Bneg = b.x - a.x;
    const double C    = a.y * Bneg - a.x * A;
    return FStandardFormLine{ A, -Bneg, C };
}
#define INTERSECTION_DET_EPSILON 0.0001f
constexpr __device__ static inline DPoint intersection(
    const FStandardFormLine l1, const FStandardFormLine l2)
{
    // ensure lines aren't parallel
    const double det = l1.a * l2.b - l2.a * l1.b;
    if (fabs(det) <= INTERSECTION_DET_EPSILON)
        return DPoint{ DBL_MAX, DBL_MAX };

    // lines aren't parallel: save intersection point
    return DPoint{ (l1.b * l2.c - l2.b * l1.c) / det,
                   (l2.a * l1.c - l1.a * l2.c) / det };
}

constexpr __device__ static inline bool isBlockingView(
    const FPoint source,
    const FPoint target,
    const FTri tri)
{
    // 1. find distance and equation from source to target
    const float dist = hypotf(target.x - source.x, target.y - source.y);
    const auto sight = eqTwoPoints(source, target);

    // 2. calculate bounding boxes of triangle edges
    const float ABminx = fminf(tri.a.x, tri.b.x);
    const float ABminy = fminf(tri.a.y, tri.b.y);
    const float ABmaxx = fmaxf(tri.a.x, tri.b.x);
    const float ABmaxy = fmaxf(tri.a.y, tri.b.y);

    const float BCminx = fminf(tri.b.x, tri.c.x);
    const float BCminy = fminf(tri.b.y, tri.c.y);
    const float BCmaxx = fmaxf(tri.b.x, tri.c.x);
    const float BCmaxy = fmaxf(tri.b.y, tri.c.y);

    const float CAminx = fminf(tri.c.x, tri.a.x);
    const float CAminy = fminf(tri.c.y, tri.a.y);
    const float CAmaxx = fmaxf(tri.c.x, tri.a.x);
    const float CAmaxy = fmaxf(tri.c.y, tri.a.y);

    // 3. rays backwards and forwards have the same equation; make sure
    //    we're on the same side of the source as the tri
    const float signax = signbit(tri.a.x  - source.x);
    const float signbx = signbit(tri.b.x  - source.x);
    const float signcx = signbit(tri.c.x  - source.x);
    const float signtx = signbit(target.x - source.x);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (signax == signbx && signbx == signcx)
    {
        // a, b, anc c are on the same x-side of source
        // return false if target is on the other side
        if (signax != signtx) return false;
    }
#pragma GCC diagnostic pop
    const float signay = signbit(tri.a.y  - source.y);
    const float signby = signbit(tri.b.y  - source.y);
    const float signcy = signbit(tri.c.y  - source.y);
    const float signty = signbit(target.y - source.y);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (signay == signby && signby == signcy)
    {
        // a, b, anc c are on the same y-side of source
        // return false if target is on the other side
        if (signay != signty) return false;
    }
#pragma GCC diagnostic pop

    // 4. calculate equations for triangle edges
    const auto AB = eqTwoPoints(tri.a, tri.b);
    const auto BC = eqTwoPoints(tri.b, tri.c);
    const auto CA = eqTwoPoints(tri.c, tri.a);

    // 5. find intersection points with sight line
    DPoint ABi = intersection(sight, AB);
    DPoint BCi = intersection(sight, BC);
    DPoint CAi = intersection(sight, CA);

    // 6. do any of those intersection points lie on the tri edge?
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (ABi.x != DBL_MAX && ABi.y != DBL_MAX)
#pragma GCC diagnostic pop
    {
        // 5. is this point within the edge bounding box?
        if (ABminx <= ABi.x && ABi.x <= ABmaxx &&
            ABminy <= ABi.y && ABi.y <= ABmaxy)
        {
            // 6. is this point closer than the target point?
            const float idist = hypotf(ABi.x - source.x, ABi.y - source.y);
            if (dist > idist) return true;
        }
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (BCi.x != DBL_MAX && BCi.y != DBL_MAX)
#pragma GCC diagnostic pop
    {
        if (BCminx <= BCi.x && BCi.x <= BCmaxx &&
            BCminy <= BCi.y && BCi.y <= BCmaxy)
        {
            const float idist = hypotf(BCi.x - source.x, BCi.y - source.y);
            if (dist > idist) return true;
        }
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (CAi.x != DBL_MAX && CAi.y != DBL_MAX)
#pragma GCC diagnostic pop
    {
        if (CAminx <= CAi.x && CAi.x <= CAmaxx &&
            CAminy <= CAi.y && CAi.y <= CAmaxy)
        {
            const float idist = hypotf(CAi.x - source.x, CAi.y - source.y);
            if (dist > idist) return true;
        }
    }
    // no edge intersections -- not blocked
    return false;
}

__global__ void render(
    uint32_t* d_buf, const uint32_t w, const uint32_t h,
    FTri* d_tris, const uint32_t ntris,
    const float actorx, const float actory, const float actorr)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t pix = y*w + x;

    if (x >= w || y >= h) return; // off-screen

    const FPoint target { (float)x, (float)y };
    const FPoint actor { actorx, actory };

    // compute this pixel's distance from actor, angle from actor, and angle
    // from actor's look angle
    float distance, angle;
    pointRelatives(actor, target, distance, angle);
    const float angleFromActorLook = fmodf(angle+actorr + M_PI, TWOPI) - M_PI;

    // is this pixel near enough to be covered by the actor?
    constexpr float actorSize = 10;
    if (distance <= actorSize) // actor draw size, in px
    {
        d_buf[pix] = colorPlayer;
        return;
    }

    // draw tris
    for (uint32_t i = 0; i < ntris; i++)
        if (withinTri(target, d_tris[i]))
        {
            d_buf[pix] = colorObstacle;
            return;
        }

    // is the actor inside a tri?
    for (uint32_t i = 0; i < ntris; i++)
        if (withinTri(actor, d_tris[i]))
        {
            d_buf[pix] = colorBackground;
            return;
        }

    // is this pixel outside the view cone?
    constexpr float viewLength = 800;
    constexpr float viewAngle = 0.6;
    if (fabs(angleFromActorLook) > viewAngle || distance > viewLength)
    {
        d_buf[pix] = colorBackground;
        return;
    }

    // is this pixel blocked by a tri?
    for (uint32_t i = 0; i < ntris; i++)
        if (isBlockingView(actor, target, d_tris[i]))
        {
            d_buf[pix] = colorBackground;
            return;
        }

    // all checks passed, visible!
    d_buf[pix] = colorVisible;
}
