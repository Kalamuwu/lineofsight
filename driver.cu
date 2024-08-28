#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <random>

#include <SDL2/SDL.h>
#include "hip-commons.h"

#include "logging.hpp"
#include "screen.hpp"
#include "rollingAverage.hpp"
#include "render.hpp"


// https://github.com/ROCm/rocm-examples/blob/b1b2122a2afa4a735e68cf4045256135d60b40a6/Common/example_utils.hpp#L179
template<typename T,
         typename U,
         std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
__host__ __device__ constexpr auto ceiling_div(
    const T& dividend, const U& divisor)
{
    return (dividend + divisor - 1) / divisor;
}


static inline void generateTris(
    FTri* tris, const std::size_t ntris,
    const float minx, const float maxx,
    const float miny, const float maxy,
    const float scale, const float areaThresh)
{
    // init RNG
    std::random_device rd;
    const uint32_t seed = (uint32_t)rd();
    std::mt19937 rng(seed);
    DEBUG("seed %u\n", seed);

    // initialize random distributions
    std::uniform_real_distribution<> unitdistrib(-1.0, 1.0);
    std::uniform_real_distribution<> xdistrib(minx, maxx);
    std::uniform_real_distribution<> ydistrib(miny, maxy);

    for (std::size_t i = 0; i < ntris; i++)
    {
        // 1. pick a random center location
        const float xcenter = xdistrib(rng);
        const float ycenter = ydistrib(rng);

        // 2. pick three random points in the square around that center point
        FTri t;
        for (FPoint* p : {&t.a, &t.b, &t.c})
        {
            const float xunit = unitdistrib(rng) * scale;
            const float yunit = unitdistrib(rng) * scale;
            p->x = xunit + xcenter;
            p->y = yunit + ycenter;
        }

        // 3. check that the generated tri has a large enough area
        const float area = std::abs(t.a.x * (t.b.y - t.c.y) +
                                    t.b.x * (t.c.y - t.a.y) +
                                    t.c.x * (t.a.y - t.b.y)) / 2.0;
        if (area < areaThresh * scale)
        {
            // reroll this tri
            i--;
            continue;
        }

        // 4. save to tris list
        tris[i] = t;
    }
}


struct Actor
{
    double x, y;
    double r;
    double speed, rotspeed;

    void moveForward(void)  { x += speed * cos(r); y -= speed * sin(r); }
    void moveBackward(void) { x -= speed * cos(r); y += speed * sin(r); }

    void rotateLeft(void)  { r = fmod(r + rotspeed + TWOPI, TWOPI); }
    void rotateRight(void) { r = fmod(r - rotspeed + TWOPI, TWOPI); }
};


int main()
{
    // initialize SDL2 and a renderable window
    Screen screen(1280, 720);
    screen.setTargetFps(50);
    uint32_t* buf = screen.getPixelBuffer();

    // set up gpu pixel buffer
    uint32_t* d_buf;
    const uint64_t bufsize =
        sizeof(uint32_t) * screen.getWidth() * screen.getHeight();
    checkHIPError(hipMalloc((void**)&d_buf, bufsize));
    checkHIPError(hipMemset(d_buf, 0x00, bufsize));
    checkHIPError(hipDeviceSynchronize());

    /// set up tris, host-side
    constexpr std::size_t ntris = 16;
    FTri* tris = new FTri[ntris];
    generateTris(
        tris, ntris,
        100, screen.getWidth()  - 100,
        100, screen.getHeight() - 100,
        60, 20);

    // set up tris, device-side
    FTri* d_tris;
    const uint64_t trissize = sizeof(FTri) * ntris;
    checkHIPError(hipMalloc((void**)&d_tris, trissize));
    checkHIPError(hipMemset(d_tris, 0x00, trissize));
    checkHIPError(hipDeviceSynchronize());

    // copy tris from host to device
    checkHIPError(hipMemcpy(d_tris, tris, trissize, hipMemcpyHostToDevice));
    checkHIPError(hipDeviceSynchronize());

    // dealloc tris, host-side
    delete[] tris;

    // movable actor to demo lines of sight
    Actor actor
    {
        screen.getWidth() / 2.0, screen.getHeight() / 2.0,
        M_PI_2, // start facing vertically up
        5, // move speed 4 px per frame
        (2.0/180.0)*M_PI // turn speed 2 degrees per frame
    };

    // prepare kernel
    const dim3 blockDims(32,32);
    const dim3 gridDims(
        ceiling_div(screen.getWidth(),  blockDims.x),
        ceiling_div(screen.getHeight(), blockDims.y));

    // fetch keyboard state pointer
    const Uint8* const keyboard = SDL_GetKeyboardState(NULL);

    // main loop
    SDL_Event e;
    while (true)
    {
        DEBUG_FILELOC();

        // handle events
        while (SDL_PollEvent(&e))
            switch (e.type)
            {
                case SDL_QUIT:
                    goto quit;

                case SDL_WINDOWEVENT:
                    screen.processEvent(e);
                    break;
            }

        // apply movement
        if (keyboard[SDL_SCANCODE_W]) actor.moveForward();
        if (keyboard[SDL_SCANCODE_A]) actor.rotateLeft();
        if (keyboard[SDL_SCANCODE_S]) actor.moveBackward();
        if (keyboard[SDL_SCANCODE_D]) actor.rotateRight();

        // do calculations
        render<<<gridDims, blockDims>>>(
            d_buf, screen.getWidth(), screen.getHeight(),
            d_tris, ntris,
            actor.x, actor.y, actor.r);
        checkHIPError(hipGetLastError());
        checkHIPError(hipDeviceSynchronize());

        // copy back to cpu and display
        checkHIPError(hipMemcpy(buf, d_buf, bufsize, hipMemcpyDeviceToHost));
        checkHIPError(hipDeviceSynchronize());
        screen.show();
    }

quit:
    DEBUG_FILELOC();
    checkHIPError(hipFree(d_buf));
    checkHIPError(hipFree(d_tris));
    return 0;
}
