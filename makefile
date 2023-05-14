# Determine the platform
OS := $(shell uname -s 2>/dev/null || (ver >nul 2>&1 && echo Windows))

# Include the appropriate platform-specific Makefile
ifeq ($(OS), Windows)
    include Makefile_Windows
else
    include Makefile_Linux
endif