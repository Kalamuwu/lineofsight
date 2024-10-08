#ifndef HIPCOMMONH
#define HIPCOMMONH

#ifndef __HIP__
#define __HIP__ 1
#endif

#ifdef __HIP_PLATFORM_NVIDIA__
#undef __HIP_PLATFORM_NVIDIA__
#endif

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip/device_functions.h"
#include "hip/math_functions.h"
#include "hiprand/hiprand_kernel.h"

#endif
