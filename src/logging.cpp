#include "logging.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>


void _checkSDLError(
    int const code,
    char const *const func,
    const char *const file,
    int const line)
{
    if (code != 0)
    {
        fprintf(stderr, "\nSDL Error: %s\n...at %s:%d '%s'\n",
            SDL_GetError(), file, line, func);
        fflush(stderr);

        #if KILL_ON_ERR
            exit(code);
        #endif
    }
}

void _checkTTFError(
    int const code,
    char const *const func,
    const char *const file,
    int const line)
{
    if (code != 0)
    {
        fprintf(stderr, "\nTTF Error: %s\n...at %s:%d '%s'\n",
            TTF_GetError(), file, line, func);
        fflush(stderr);

        #if KILL_ON_ERR
            exit(code);
        #endif
    }
}


void _checkHIPError(
    hipError_t const code,
    char const *const func,
    const char *const file,
    int const line)
{
    if (code != 0)
    {
        fprintf(stderr, "\nHIP Error: %s\n...at %s:%d '%s'\n",
            hipGetErrorString(code), file, line, func);
        fflush(stderr);

        #if KILL_ON_ERR
            hipDeviceReset();
            exit(static_cast<unsigned int>(code));
        #endif
    }
}
