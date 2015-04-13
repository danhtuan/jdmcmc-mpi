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

double likelihood(const CImg<unsigned char> &image,
  const CImg<unsigned char> &target, vector<Point> Uxy, int K) {

  int x = image.height();
  int y = image.width();

  CImg<unsigned char> It(128,128); // XXX: warning, hardcoded image dims here
  It.fill(0);

  // draw our target points on the image
  for (int i=0; i<K; i++) {
    It(Uxy[i].x, Uxy[i].y) = 1;
  }

  // convolve the target circle with the points (basically draws the target
  // at each point), then threshold the image, normalize black to 129, white
  // to 193, and invert so target areas are 63 and non-target areas are 127
  CImg<unsigned char> Ie = -It
    .get_convolve(target, 0, false)
    .get_threshold(128)
    .get_normalize(129,193);

  double mse = image.MSE(Ie);

  printf("MSE %.3f\n", mse);

  return exp(-0.5*mse);
}

int main() {
  CImg<unsigned char> image("images/discs20.bmp"), imtemp;
  CImg<unsigned char> target("images/target.bmp");
  boost::math::poisson_distribution<> pd(lambda);

  float white[3] = {255,255,255};

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
  boost::uniform_int<> jump(0,2);

  imtemp = image;
  for (int i=0; i < num_objs[0]; i++) {
    Oxy[i].x = randrow(rng);
    Oxy[i].y = randcol(rng);
    imtemp.draw_circle(Oxy[i].x, Oxy[i].y, 9, white, 0.9f, 1);
  }
  // show figure
  imtemp.display(main_disp, 0);

  obj_fn[0] = likelihood(image, target, Oxy, num_objs[0])
                * pdf(pd,num_objs[0]);

  for (int i=1; i<sampling_steps; i++) {
    // start time
    int a=jump(rng);
    printf("Iteration[%02u]--a:[%d]", i, a);
    if (a==0 && num_objs[i-1] > 1) {
      printf("--Jump-1");
      num_objs[i] = num_objs[i-1] - 1;
    } else if (a==1 && num_objs[i-1] < max_objects) {
      printf("--Jump+1");
      num_objs[i] = num_objs[i-1] + 1;
    } else {
      printf("--Jump+0");
      num_objs[i] = num_objs[i-1];
    }
    printf("--Discs:[%02u]\n", num_objs[i]);

    // gibbs sampling
    // accept/reject
    // obj_fn[i] = likelihood() * poisspdf();

    pa_jump = obj_fn[i]/obj_fn[i-1];
  }

  printf("\nProgram Exit\n");
  return 0;
}
