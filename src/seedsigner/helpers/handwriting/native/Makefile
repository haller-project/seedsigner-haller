# Shared library loaded through ctypes by native/__init__.py: required by the
# handwriting reader. SeedSigner OS cross-compiles it with the image.
# On a Raspberry Pi Zero:  make CFLAGS_ARCH="-mcpu=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard"
# Portable C instead of the ARM SMLAD instruction:  add -DHW_GENERIC to CFLAGS_ARCH
CFLAGS = -O3 -funroll-loops -ffp-contract=off -std=c99 -Wall -Wextra -fPIC $(CFLAGS_ARCH)

libhandwriting.so: handwriting.c handwriting.h
	$(CC) $(CFLAGS) -shared -o $@ handwriting.c -lm

.PHONY: clean
clean:
	rm -f libhandwriting.so
