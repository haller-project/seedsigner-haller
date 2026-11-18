# Shared library loaded by word_model.py and decoder.py through ctypes.
# On the Raspberry Pi Zero:  make CFLAGS_ARCH="-mcpu=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard"
CFLAGS = -O3 -funroll-loops -ffp-contract=off -std=c99 -Wall -Wextra -fPIC $(CFLAGS_ARCH)

libhandwriting.so: handwriting.c handwriting.h
	$(CC) $(CFLAGS) -shared -o $@ handwriting.c -lm

.PHONY: clean
clean:
	rm -f libhandwriting.so
