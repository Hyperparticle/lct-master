CC=gcc
CFLAGS=-Wall -g -O -I.
LFLAGS=

PROGS=splaygen splaytree

all: $(PROGS)

splaygen: splaygen.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

splaytree: node.o operation.o tree.o splaytree.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

clean:
	rm *.o splaygen splaytree
