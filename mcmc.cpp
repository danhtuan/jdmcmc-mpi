#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

// rng & normal/poisson distribution
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
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

CImgDisplay main_disp(128,128,"Main",0);
unsigned char white[1] = {255};

struct Point {
  int x,y;
};

typedef CImg<unsigned char> Img; 

double likelihood(const Img &image,
  const Img &target, vector<Point> Uxy, int K) {

  int x = image.height();
  int y = image.width();

  Img Ie(128,128,1,1,0); // XXX: warning, hardcoded image dims here

  for (auto it = Uxy.begin(); it != Uxy.end(); it++) {
    Ie.draw_image(it->x-9, it->y-9, target, target, 255);
  }
  Ie = -Ie.normalize(129,193);
  
  double mse = image.MSE(Ie); // note: gibbs is sensitive to this
  double like = exp(-0.5*mse);

  return like;
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

// uniform [0,1) distribution double
double random_uniform() {
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
    uni_dist(gen, boost::uniform_real<>(0,1));
  return uni_dist();
}

// normal (gaussian, 0 mean, 1 stdev)
double random_normal() {
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
    normal(gen, boost::normal_distribution<>(0.0,1.0));
  return normal();
}

vector<Point> gibbs_sampling(const Img &image,
  const Img &target, int num_objs) {
  Img imtemp;
  int T=50, K_MAX = 50;
  int rows = image.width();
  int cols = image.height();

  printf("/n\t\tGibbs:[Step/Discs-100/25]");

  vector<vector<Point>> AOxy (T);

  for (int i=0; i < num_objs; i++) {
    Point point = gen_random_point(rows,cols);
    AOxy[0].push_back(point);
  }
  vector<Point> Cur_Oxy = AOxy[0];
  double L1 = likelihood(image,target,Cur_Oxy,num_objs);

  for (int t=1; t<T; t++) {
    for (int i=0; i<num_objs; i++) {
      printf("%03u/%02u\n", t, i);
      Point Oxy = Cur_Oxy[i];
      for (int j=0; j<K_MAX; j++) {
        Point Dxy = Oxy;
        // note: matlab version uses round()
        Dxy.x += int(random_normal()*20);
        Dxy.y += int(random_normal()*20);
        Dxy = clamp(Dxy,rows,cols);
        vector<Point> New_Cur_Oxy = Cur_Oxy;
        New_Cur_Oxy[i] = Dxy;
        double L2 = likelihood(image,target,New_Cur_Oxy,num_objs);
        double v = min(1.0,L2/L1);
        double u = random_uniform();
        if (v > u) {
          Oxy = Dxy;
          Cur_Oxy = New_Cur_Oxy;
          L1 = L2;
        }
      }
      AOxy[t] = Cur_Oxy;

      // show image
      imtemp = image;
      for (int n=0; n < num_objs; n++) {
        imtemp.draw_circle(Cur_Oxy[n].x, Cur_Oxy[n].y, 9, white, 0.9f, 1);
      }
      imtemp.display(main_disp);
    }
  }
  printf("Done Gibbs");
  return Cur_Oxy;
}

int main(int argc, char *argv[]) {
  Img image_load("images/discs20.bmp"), image, imtemp;
  Img target_load("images/target.bmp"), target;
  image = image_load.channel(0);
  target = target_load.channel(0);
  boost::math::poisson_distribution<> pd(lambda);

  int rows = image.height();
  int cols = image.width();

  printf("Image is %d x %d\n", cols, rows);

  vector<int> num_objs (sampling_steps);
  num_objs[0] = init_k;

  vector<double> obj_fn (sampling_steps);

  // init rng
  boost::mt19937 rng;
  boost::uniform_int<> jump(0,2);

  imtemp = image;
  vector<vector<Point>> Oxy (sampling_steps);
  for (int i=0; i < num_objs[0]; i++) {
    Point point = gen_random_point(rows,cols);
    Oxy[0].push_back(point);
    imtemp.draw_circle(point.x, point.y, 9, white, 0.9f, 1);
  }
  // show figure
  imtemp.display(main_disp);

  obj_fn[0] = likelihood(image, target, Oxy[0], num_objs[0])
                * pdf(pd,num_objs[0]);

  printf("Iteration[%02u]--Discs:[%02u]--OBJ_FN:[%.5e]--Duration[%5.5f]\n", 1, num_objs[0], obj_fn[0], 0.0);

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

    Oxy[i] = gibbs_sampling(image,target,num_objs[i]);
    // accept/reject
    obj_fn[i] = likelihood(image, target, Oxy[i], num_objs[i])
                * pdf(pd,num_objs[i]);
    double pa_jump = min(obj_fn[i]/obj_fn[i-1], 1.0);
    printf("\n\t\tOBJ_FN:[%d]--AcceptRate:[%2.2f]",obj_fn[i], pa_jump);
    if (pa_jump > random_uniform()) {
      printf("--Accept");
      // add video frame ?
    } else {
      printf("--Reject");
      // duplicate previous step
      num_objs[i] = num_objs[i-1];
      Oxy[i] = Oxy[i-1];
      obj_fn[i] = obj_fn[i-1];
    }
    //printf("--Duration:[%3.3f]\n", timeblah);
    printf("\n");
  }
  // step 3 - burn in
  printf("Burning to get Experimental Result...\n");
  // step 4 - mean estimate of object number k*
  printf("\nProgram Exit\n");
  return 0;
}
