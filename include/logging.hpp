#ifndef LOGGINGH
#define LOGGINGH

#include <stdio.h>

#include "hip-commons.h"  // hipError_t


#define PRINT_DEBUGS true
#define DEBUG_GENERALS true
#define DEBUG_FILELOCS false

#define KILL_ON_ERR true


// error checking for SDL calls
#define checkSDLError(val) \
    _checkSDLError( (val), #val, __FILE__, __LINE__)
void _checkSDLError(
    int const code,
    char const *const func,
    const char *const file,
    int const line);

// error checking for TTF calls
#define checkTTFError(val) \
    _checkTTFError( (val), #val, __FILE__, __LINE__)
void _checkTTFError(
    int const code,
    char const *const func,
    const char *const file,
    int const line);

// error checking for HIP calls
#define checkHIPError(val) \
    _checkHIPError( (val), #val, __FILE__, __LINE__)
void _checkHIPError(
    hipError_t const code,
    char const *const func,
    const char *const file,
    int const line);

// general logging
#if DEBUG_GENERALS and PRINT_DEBUGS
#define DEBUG(...) do {                                                       \
    fprintf(stderr, __VA_ARGS__);                                             \
    fflush(stderr);                                                           \
} while (0)
#else
#define DEBUG(...)
#endif
/*
// general logging
#if DEBUG_GENERALS and PRINT_DEBUGS
#define DEBUG(...) do {                                                       \
    fprintf(stderr, __FILE__ ":%d\t! debug: ", __LINE__);                     \
    fprintf(stderr, __VA_ARGS__);                                             \
    fflush(stderr);                                                           \
} while (0)
#else
#define DEBUG(...)
#endif
*/

// file location logging
#if DEBUG_FILELOCS and PRINT_DEBUGS
#define DEBUG_FILELOC() do {                                                  \
    fprintf(stderr, "  " __FILE__ ":%d\n", __LINE__);                         \
    fflush(stderr);                                                           \
} while (0)
#else
#define DEBUG_FILELOC()
#endif

#endif // LOGGINGH
