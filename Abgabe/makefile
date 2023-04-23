#
# This makefile does many thing by convention which means
# less fiddling arround with the makefile.
#
# Written by Sebastian Bauer <mail@sebastianbauer.info> for
# lectures at the HTW.
#
# Edit a little bit for the Project, but works mostly the same.
#
# This makefile is released in public domain.
#

# The name of the main executable
TARGET=net-exe

# Flags and stuff, change as required
OPTIMIZE=-O3
# Flags for all languages
CPPFLAGS=-ggdb $(OPTIMIZE) -Wall -MMD -MP
# Flags for C only
CFLAGS=-Wmissing-prototypes -fPIC
# Flags for the linker
LDFLAGS=
# Additional linker libs
LDLIBS=-lm

#libary objects
SH_OBJS=src/netPy.so
#SHAREE
SH=-shared

# Compilers
CC=gcc -fopenmp

##############################################################
# Normally there shouldn't be a need to modify the following
# lines, expect for bug fixing and new features.

# The sources that make up the main executable. These are
# all files that end with an .c but do not end with _test.c
SRCS=$(wildcard */*.c)

# We make up the objects by replacing the .c and .cpp suffixes
# with a .o suffix
OBJS=$(patsubst %.c,%.o,$(SRCS))

# The dependency files
DEPS=$(SRCS:.c=.d)

# The first target (all) is allways the default target
.PHONY: all
all: build pybuild

# Our buld target depends on the real target
.PHONY: build
build: $(TARGET)
# Our target is built up from the objects
$(TARGET): $(OBJS)

.PHONY: pybuild
pybuild: $(SH_OBJS)

$(SH_OBJS): $(OBJS)
	$(CC) $(SH) -o $@ $^

clean:
	rm -Rf $(OBJS) $(SH_OBJS) $(TARGET) $(DEPS)

-include $(DEPS)
