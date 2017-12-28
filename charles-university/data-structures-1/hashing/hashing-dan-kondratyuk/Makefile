CC=gcc
CFLAGS=-std=c99 -Wall -O3
LFLAGS=-lm

PROGS=hashing

all: $(PROGS)

hashing: main.o hash-scheme.o hash-system.o random-gen.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

clean:
	rm -f *.o $(PROGS)