#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

// rng & normal/poisson distribution
#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
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

// XXX: make this selectable via command line arguments
boost::mt19937 gen;
// boost::mt19937 gen(time(0)); // seeded version

struct Point {
  int x,y;
};

double likelihood(const CImg<unsigned char> &image,
  const CImg<unsigned char> &target, vector<Point> Uxy, int K) {

  int x = image.height();
  int y = image.width();

  CImg<unsigned char> It(128,128); // XXX: warning, hardcoded image dims here
  It.fill(0);

  // draw our target points on the image
  for (auto it = Uxy.begin(); it != Uxy.end(); it++) {
    It(it->x, it->y) = 1;
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

Point gen_random_point(int width, int height) {
  using namespace boost;

  variate_generator<mt19937&, uniform_int<> >
    randcol(gen, uniform_int<>(0,width-1));
  variate_generator<mt19937&, uniform_int<> >
    randrow(gen, uniform_int<>(0,height-1));

  return Point{randrow(), randcol()};
}

int clamp(int x, int min, int max) {
  return (x<min) ? min : (x>max) ? max : x; 
}

Point clamp(Point p, int w, int h) {
  p.x = clamp(p.x, 0, w-1);
  p.y = clamp(p.y, 0, h-1);
  return p; 
}

double random_normal() {
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
    normal(gen, boost::normal_distribution<>(0.0,1.0));
  return normal();
}

void gibbs_sampling(const CImg<unsigned char> &image,
  const CImg<unsigned char> &target, int num_objs) {
  int T=50, M_BURN_IN = 50, N_BURN_IN = 2, K_MAX = 50;
  int rows = image.width();
  int cols = image.height();

  printf("\n\t\tGibbs:[Step/Discs-100/25]");

  // want AOxy = num_objs*x * T
  // random points in first set of AOxy
  // Cur_Oxy = first AOxy
  // show pic here maybe
  vector<Point> AOxy (T), Cur_Oxy;
  double L1 = likelihood(image,target,AOxy,num_objs);

  for (int t=1; t<T; t++) {
    for (int i=0; i<num_objs; i++) {
      printf("\b\b\b\b\b\b\b%03u/%02u", t, i);
      Point Oxy;// = Cur_Oxy[i];
      Oxy.x = 64; Oxy.y = 64;
      for (int j=0; j<K_MAX; j++) {
        Point Dxy = Oxy;
        // note: matlab version uses round()
        Dxy.x += int(random_normal()*20);
        Dxy.y += int(random_normal()*20);
        Dxy = clamp(Dxy,rows,cols);
        //printf("%d %d\n",Dxy.x,Dxy.y);
      }
    }
  }
}

int main(int argc, char *argv[]) {
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
  double pa_jump;

  // init rng
  boost::mt19937 rng;
  boost::uniform_int<> jump(0,2);

  imtemp = image;
  vector<Point> Oxy;
  for (int i=0; i < num_objs[0]; i++) {
    Point point = gen_random_point(rows,cols);
    Oxy.push_back(point);
    imtemp.draw_circle(point.x, point.y, 9, white, 0.9f, 1);
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
    printf("--Discs:[%02u]", num_objs[i]);

    gibbs_sampling(image,target,num_objs[i]);
    // accept/reject
    // obj_fn[i] = likelihood() * poisspdf();

    pa_jump = obj_fn[i]/obj_fn[i-1];
  }

  printf("\nProgram Exit\n");
  return 0;
}
