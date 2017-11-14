CC=gcc
CFLAGS=-Wall -g -O -I.
LFLAGS=

PROGS=heapgen fibheap

all: $(PROGS)

heapgen: heapgen.o $(OBJECTS)
	g++ $(CFLAGS) -o $@ $^ $(LFLAGS)

fibheap: node.o operation.o heap.o fibheap.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

clean:
	rm -f *.o $(PROGS)
