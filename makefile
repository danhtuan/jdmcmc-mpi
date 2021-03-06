CC=mpic++
CFLAGS=-O3 -std=c++0x -lm -lX11 -lpthread

all: mcmc 

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ -c $<

mcmc: mcmc.o
	$(CC) $(CFLAGS) -o mcmc mcmc.o

ci: create_image.o
	$(CC) $(CFLAGS) -o ci create_image.o

clean:
	rm mcmc ci *.o 

gibbs: gibbs.o
	$(CC) $(CFLAGS) -o gibbs gibbs.o
