#include "screen.hpp"

#include <stdio.h>
#include <memory>

#define FONTPATH "/usr/share/fonts/noto/NotoSansMono-Regular.ttf"


Screen::Screen(const uint32_t w, const uint32_t h) : m_width(w), m_height(h)
{
    // init SDL2
    constexpr Uint32 initFlags =
        SDL_INIT_TIMER | SDL_INIT_VIDEO | SDL_INIT_EVENTS;
                                          // ah, SDL2...
    checkSDLError(SDL_Init(initFlags));      // >0 on fail
    checkTTFError(TTF_Init(/* no flags*/));  // -1 on fail
    // SDL_gfx needs no init

    // create the window
    //constexpr Uint32 windowFlags = SDL_WINDOW_RESIZABLE;
    constexpr Uint32 windowFlags = 0;
    mp_window = SDL_CreateWindow(
        "LineOfSight Demo",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        w, h,
        windowFlags);
    //SDL_SetWindowMinimumSize(mp_window, 800, 700);
    checkSDLError(mp_window == NULL);

    // create the renderer
    constexpr Uint32 renderFlags = SDL_RENDERER_ACCELERATED;
    mp_renderer = SDL_CreateRenderer(mp_window, -1, renderFlags);
    checkSDLError(mp_renderer == NULL);
    // SDL_BlendMode blend = SDL_BlendMode::SDL_BLENDMODE_BLEND;
    // checkSDLError(SDL_SetRenderDrawBlendMode(mp_renderer, blend));

    // create the texture that we will draw on
    mp_display = SDL_CreateTexture(
        mp_renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STATIC,
        m_width, m_height);
    checkSDLError(mp_display == NULL);

    // ...and its custom pixel buffer
    mp_textureBuffer = new uint32_t[ m_width * m_height ];

    // create our font
    mp_font = TTF_OpenFont(FONTPATH, 12);
    checkTTFError(mp_font == NULL);

    // for fps tracking+limiting
    const Uint32 ticksNow = SDL_GetTicks();
    m_fpsData.framesSinceLastUpd = 0;
    m_fpsData.lastFrameTicks = ticksNow;
    m_fpsData.lastUpdTicks = ticksNow;

    // all OK!
    DEBUG("screen: initialized OK\n");
}

Screen::~Screen()
{
    DEBUG_FILELOC();

    delete[] mp_textureBuffer;
    mp_textureBuffer = nullptr;

    TTF_CloseFont(mp_font);
    mp_font = nullptr;

    SDL_DestroyTexture(mp_display);
    mp_display = nullptr;

    SDL_DestroyRenderer(mp_renderer);
    mp_renderer = nullptr;

    SDL_DestroyWindow(mp_window);
    mp_window = nullptr;

    // quit SDL2 subsystems
    TTF_Quit();
    SDL_Quit();

    DEBUG("screen: quit OK\n");
}


void Screen::processEvent(SDL_Event& e)
{
    DEBUG_FILELOC();

    // handle window events here
    if (e.type == SDL_WINDOWEVENT)
    {
        switch(e.window.event)
        {
            case SDL_WINDOWEVENT_SIZE_CHANGED:
                checkSDLError(SDL_RenderSetLogicalSize(
                    mp_renderer,
                    e.window.data1, e.window.data2));
                m_width  = e.window.data1;
                m_height = e.window.data2;
                DEBUG("screen: event: window resized, new res %d,%d\n",
                    m_width, m_height);
                break;

            case SDL_WINDOWEVENT_MINIMIZED:
                setMinimized(true);
                DEBUG("screen: event: window minimized\n");
                break;

            case SDL_WINDOWEVENT_MAXIMIZED:
            case SDL_WINDOWEVENT_RESTORED:
                setMinimized(false);
                DEBUG("screen: event: window no longer minimized\n");
                break;
        }
    }
}


void Screen::clampToScreen(Point& p) const
{
    // compiler will complain that we're checking sint > uint. since we
    // already checked the sint is >= 0, we don't need to worry about the
    // difference in signs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
    if (p.x < 0) p.x = 0;
    else if (p.x > m_width) p.x = m_width;
    if (p.y < 0) p.y = 0;
    else if (p.y > m_height) p.y = m_height;
#pragma GCC diagnostic pop
}


void Screen::show()
{
    DEBUG_FILELOC();

    // show fps in top-left
    char s[8] {0};
    sprintf(s, "FPS %3u", std::min(std::max(m_fps, 0u), 999u));
    drawText(s, Point{ 4,4 }, Color{ 0xFFFFFFFF });

    // copy display texture to window and render
    checkSDLError(SDL_UpdateTexture(
        mp_display, NULL, mp_textureBuffer, m_width * sizeof(uint32_t)));
    checkSDLError(SDL_RenderCopy(mp_renderer, mp_display, NULL, NULL));
    SDL_RenderPresent(mp_renderer);

    // calculate fps
    const Uint32 ticksNow = SDL_GetTicks();
    m_fpsData.framesSinceLastUpd++;
    if (ticksNow - m_fpsData.lastUpdTicks > 1000)
    {
        m_fps = m_fpsData.framesSinceLastUpd;
        m_fpsData.framesSinceLastUpd = 0;
        m_fpsData.lastUpdTicks = ticksNow;
    }

    // calculate delay to reach target fps
    if (m_fpsData.targetFps)
    {
        const Uint32 ticksNow = SDL_GetTicks();
        const Uint32 targetDelay = 1000 / m_fpsData.targetFps;
        const Uint32 ticksElapsed = ticksNow - m_fpsData.lastFrameTicks;
        if (ticksElapsed > targetDelay)
            DEBUG("Can't keep up! Running %d ms behind "
                  "(target delay %d ms, elapsed %d ms)\n",
                ticksElapsed - targetDelay,
                targetDelay, ticksElapsed);
        else SDL_Delay(targetDelay - ticksElapsed);
        m_fpsData.lastFrameTicks = SDL_GetTicks();
    }
}


// standard getters
uint32_t Screen::getWidth()     const { return m_width;             }
uint32_t Screen::getHeight()    const { return m_height;            }
uint32_t Screen::getFps()       const { return m_fps;               }
uint32_t Screen::getTargetFps() const { return m_fpsData.targetFps; }

// standard setters
void Screen::setTargetFps(uint32_t target) { m_fpsData.targetFps = target; }

// getters and setters for m_flags
static constexpr uint8_t minimizedFlag = 0x01;

bool Screen::isMinimized() const { return m_flags & minimizedFlag; }
void Screen::setMinimized(const bool b)
{
    if (b) m_flags |=  minimizedFlag;
    else   m_flags &= ~minimizedFlag;
}


// primitive drawing functions

uint32_t* Screen::getPixelBuffer() const
{
    return mp_textureBuffer;
}

void Screen::clear(const Color c)
{
    const uint64_t n = m_width * m_height;
    for (uint64_t i = 0; i < n; i++)
        mp_textureBuffer[i] = c.col();
}

void Screen::pixel(const Point p, const Color c)
{
    // compiler will complain that we're checking sint > uint. since we
    // already checked the sint is >= 0, we don't need to worry about the
    // difference in signs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
    if (p.x < 0) return;
    else if (p.x > m_width) return;
    if (p.y < 0) return;
    else if (p.y > m_height) return;
#pragma GCC diagnostic pop

    mp_textureBuffer[p.y*m_width + p.x] = c.col();
}


// specialized drawing functions

void Screen::drawText(const char* s, const Point topLeft, const Color c)
{
    SDL_Surface* surf = TTF_RenderText_Solid(mp_font, s, {255,255,255,255});
    checkTTFError(surf == NULL);
    if ((surf != NULL) && (surf->w > 0) && (surf->h > 0))
    {
        SDL_LockSurface(surf);
        Uint8* const pixels = (Uint8*)surf->pixels;
        // blit each pixel to screen
        for (int32_t y = 0; y < surf->h; y++)
            for (int32_t x = 0; x < surf->w; x++)
                if (pixels[y * surf->pitch + x])
                    pixel(Point{ topLeft.x+x, topLeft.y+y }, c);
        SDL_UnlockSurface(surf);
    }
    // safe to call SDL_FreeSurface with NULL
    SDL_FreeSurface(surf);
}
