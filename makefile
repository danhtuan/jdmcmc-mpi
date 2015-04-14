debug: mcmc.cpp
	g++ -std=c++0x -lm -lX11 -lpthread -o mcmc mcmc.cpp

mcmc: mcmc.cpp
	g++ -O2 -std=c++0x -lm -lX11 -lpthread -o mcmc mcmc.cpp

ci: create_image.cpp
	g++ -O2 -std=c++0x -lm -lX11 -lpthread -o ci create_image.cpp

all: mcmc 

clean:
	rm mcmc ci
