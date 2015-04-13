debug: mcmc.cpp
	g++ -lm -lX11 -lpthread -o mcmc mcmc.cpp

mcmc: mcmc.cpp
	g++ -O2 -lm -lX11 -lpthread -o mcmc mcmc.cpp

all: mcmc 

clean:
	rm mcmc
