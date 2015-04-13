#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

// rng & poisson distribution
#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions/poisson.hpp>

// image processing & display
#include "CImg.h"

using namespace std;
using namespace cimg_library;

int lambda = 20;
int sampling_steps = 20;
int max_objects = 25;
int init_k = 19;
int M_BURN_IN = 3;
int STEP_BURN_IN = 2;

class Point {
  public: int x,y;
};

double likelihood(const CImg<unsigned char> &image, const CImg<unsigned char> &target, vector<Point> Uxy, int K) {
  int x = image.height();
  int y = image.width();

  CImg<unsigned char> It(128,128);
  It.fill(0);

  for (int i=0; i<K; i++) {
    It(Uxy[i].x, Uxy[i].y) = 1;
  }

  CImg<unsigned char> Ie=-It.get_convolve(target, 0, false).get_threshold(128).get_normalize(129,193);

  double mse = image.MSE(Ie);

  printf("MSE %.3f\n", mse);

  return exp(-0.5*mse);
}

int main() {
  CImg<unsigned char> image("images/discs20.bmp"), imtemp;
  CImg<unsigned char> target("images/target.bmp");
  boost::math::poisson_distribution<> pd(lambda);

  float white[3] = {255,255,255};

  imtemp = image;

  CImgDisplay main_disp(image,"Figure 1");

  int rows = image.height();
  int cols = image.width();

  printf("Image is %d x %d\n", cols, rows);

  vector<int> num_objs (sampling_steps);
  num_objs[0] = init_k;

  vector<double> obj_fn (sampling_steps);
  vector<Point> Oxy (num_objs[0]);
  double pa_jump;

  // init rng
  boost::mt19937 rng;
  boost::uniform_int<> randrow(0,rows-1); // [0,rows-1]
  boost::uniform_int<> randcol(0,cols-1); // [0,cols-1]

  for (int i=0; i < num_objs[0]; i++) {
    Oxy[i].x = randrow(rng);
    Oxy[i].y = randcol(rng);
    imtemp.draw_circle(Oxy[i].x, Oxy[i].y, 9, white, 0.9f, 1);
  }
  imtemp.display(main_disp, 0);

  // show figure
  obj_fn[0] = likelihood(image, target, Oxy, num_objs[0]) * pdf(pd,num_objs[0]);

  for (int i=1; i<sampling_steps; i++) {
    // start time
    double a=1.0;// = rand(1); // (0,1)
    printf("Iteration[%02u]--a:[%1.2f]", i, a);
    if (a<0.33 && num_objs[i-1] > 1) {
      printf("--Jump-1");
      num_objs[i] = num_objs[i-1] - 1;
    } else if (a<0.66 && num_objs[i-1] < max_objects) {
      printf("--Jump+1");
      num_objs[i] = num_objs[i-1] + 1;
    } else {
      printf("--Jump+0");
      num_objs[i] = num_objs[i-1];
    }
    printf("--Discs:[%02u]\r", num_objs[i]);

    // gibbs sampling
    // accept/reject
    // obj_fn[i] = likelihood() * poisspdf();

    pa_jump = obj_fn[i]/obj_fn[i-1];
  }

  printf("\nProgram Exit\n");
  return 0;
}
