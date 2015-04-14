debug: mcmc.cpp
	g++ -std=c++0x -lm -lX11 -lpthread -o mcmc mcmc.cpp

mcmc: mcmc.cpp
	g++ -O2 -std=c++0x -lm -lX11 -lpthread -o mcmc mcmc.cpp

all: mcmc 

clean:
	rm mcmc
