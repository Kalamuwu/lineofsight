CC_PATH      ?= /opt/rocm
CC            = $(CC_PATH)/bin/hipcc

BUILD_DIR    ?= build

LIBRARIES     = -lSDL2 -lSDL2_ttf
INCLUDES      = -I/opt/rocm/include -Iinclude -Ilib

# select one of these for Debug vs. Release
#CC_DBG        = -g -ggdb3
CC_DBG        =

CCWARN        = -Wall -Wextra -Wpedantic -Wfloat-equal -Wpointer-arith -Wunreachable-code -Wmissing-field-initializers

#CCOPTIM       = -O2
CCOPTIM       = -O3 -ffast-math

CCARCH        = -m64
#CCARCH        = -march=x86-64 -mtune=generic
#CCARCH        = -march=znver4 -mtune=znver4

CCFLAGS       = $(CC_DBG) $(INCLUDES) -std=c++17 $(CCARCH) $(CCOPTIM) $(CCWARN)
LDFLAGS       = $(CC_DBG) $(LIBRARIES)


# MAIN EXECUTABLE
EXECUTABLE = hip-lineofsight
SOURCES = driver.cu $(shell find src -name "*.c*")
OBJECTS_C = $(SOURCES:.c=.o)
OBJECTS_C_CPP = $(OBJECTS_C:.cpp=.o)
OBJECTS_C_CPP_CU = $(OBJECTS_C_CPP:.cu=.o)
OBJECTS = $(addprefix ${BUILD_DIR}/,$(OBJECTS_C_CPP_CU))


.PHONY: main
.DEFAULT: main

main: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $(BUILD_DIR)/$(EXECUTABLE) $(OBJECTS)

$(BUILD_DIR)/%.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@
$(BUILD_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@
$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*
