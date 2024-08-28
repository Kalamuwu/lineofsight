#ifndef SCREENH
#define SCREENH

#include <stdint.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "logging.hpp"

/**
 * `Point`
 * Represents a point on the screen.
 */
struct Point
{
    int32_t x, y;
};

/**
 * `Color`
 * Represents a color of a pixel.
 */
struct Color
{
    constexpr Color(
        const uint8_t r,
        const uint8_t g,
        const uint8_t b,
        const uint8_t a);
    constexpr Color(const uint32_t col);

    constexpr uint8_t r(void) const;
    constexpr uint8_t g(void) const;
    constexpr uint8_t b(void) const;
    constexpr uint8_t a(void) const;

    constexpr uint32_t col(void) const;

private:
    uint8_t m_col[4];
};

// Color functions: defined here for constexpr support

#if __BYTE_ORDER == __LITTLE_ENDIAN
constexpr Color::Color(
    const uint8_t r,
    const uint8_t g,
    const uint8_t b,
    const uint8_t a)
:
    m_col{ a, b, g, r }
{}
constexpr Color::Color(const uint32_t col)
:
    m_col{
        (uint8_t)(col >>  0),
        (uint8_t)(col >>  8),
        (uint8_t)(col >> 16),
        (uint8_t)(col >> 24)}
{}
constexpr uint8_t Color::r(void) const { return m_col[3]; }
constexpr uint8_t Color::g(void) const { return m_col[2]; }
constexpr uint8_t Color::b(void) const { return m_col[1]; }
constexpr uint8_t Color::a(void) const { return m_col[0]; }
constexpr uint32_t Color::col(void) const
{ return ((r()<<24) | (g()<<16) | (b()<<8) | a()); }
#elif __BYTE_ORDER == __BIG_ENDIAN
constexpr Color::Color(
    const uint8_t r,
    const uint8_t g,
    const uint8_t b,
    const uint8_t a)
:
    m_col{ r, g, b, a }
{}
constexpr Color::Color(const uint32_t col)
:
    m_col{
        (uint8_t)(col >> 24),
        (uint8_t)(col >> 16),
        (uint8_t)(col >>  8),
        (uint8_t)(col >>  0)}
{}
constexpr uint8_t Color::r(void) const { return m_col[0]; }
constexpr uint8_t Color::g(void) const { return m_col[1]; }
constexpr uint8_t Color::b(void) const { return m_col[2]; }
constexpr uint8_t Color::a(void) const { return m_col[3]; }
constexpr uint32_t Color::col(void) const
{ return ((a()<<24) | (b()<<16) | (g()<<8) | r()); }
#else
# error "Please fix <bits/endian.h>"
#endif


/**
 * Screen
 * Wrapper class around SDL2 implementation details
 */
class Screen
{
public:
    Screen(const uint32_t width, const uint32_t height);
    ~Screen();

    void processEvent(SDL_Event& e);
    void show(void);

    // helper function to clamp a point to the screen
    void clampToScreen(Point& p) const;

    // primitive drawing functions
    uint32_t* getPixelBuffer(void) const;
    void clear(             const Color);
    void pixel(const Point, const Color);

    // specialized drawing functions
    void drawText(
        const char* string,
        const Point topLeft,
        const Color color);

    // getters
    uint32_t getWidth(void) const;
    uint32_t getHeight(void) const;
    uint32_t getFps(void) const;
    uint32_t getTargetFps(void) const;

    // setters
    void setTargetFps(uint32_t target);

    // getters for bools within m_flags
    bool isMinimized(void) const;
    bool isMouseOver(void) const;

protected:
    // setters for bools within m_flags
    inline void setMinimized(const bool);
    inline void setMouseOver(const bool);

private:
    uint32_t m_width, m_height;
    uint32_t* mp_textureBuffer;

    // for tracking fps
    struct {
        uint32_t targetFps = 0;
        uint32_t framesSinceLastUpd = 0;
        uint32_t lastFrameTicks;
        uint32_t lastUpdTicks;
    } m_fpsData;
    uint32_t m_fps = 0;

    // flags representing bools about the screen
    uint32_t m_flags = 0;

    // SDL2 render stuffs
    SDL_Renderer* mp_renderer = nullptr;
    SDL_Window* mp_window = nullptr;
    SDL_Texture* mp_display = nullptr;
    TTF_Font* mp_font = nullptr;
};


#endif // SCREENH
