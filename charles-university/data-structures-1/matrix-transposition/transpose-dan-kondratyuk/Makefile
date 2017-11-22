CC=gcc
CFLAGS=-std=c99 -Wall -O3
LFLAGS=-lm

PROGS=cachesim matrix matrix-print

all: $(PROGS)

cachesim: cachesim.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

matrix: main.o matrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

matrix-print: main.o matrix.o
	$(CC) $(CFLAGS) -o $@ $^ -DPRINT_SWAP $(LFLAGS)

clean:
	rm -f *.o $(PROGS)