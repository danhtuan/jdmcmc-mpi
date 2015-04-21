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
int init_k;
int M_BURN_IN = 3;
int STEP_BURN_IN = 2;

int width;
int height;

// XXX: make this selectable via command line arguments ?
boost::mt19937 gen;
// boost::mt19937 gen(time(0)); // seeded version

CImgDisplay main_disp;
double white[1] = {1.0};

struct Point {
  int x,y;
};

typedef CImg<double> Img; 
Img black;

double likelihood(const Img &image,
  const Img &target, vector<Point> Uxy, int K) {

  Img Ie(width,height,1,1,0);

  for (auto it = Uxy.begin(); it != Uxy.end(); it++) {
    Ie.draw_image(it->x-9, it->y-9, target, target, 1.0);
  }

  Ie = (image + Ie*0.25)-0.5;

  double mse = Ie.MSE(black)*width*height; // note: gibbs is sensitive to this
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

  printf("\n\t\tGibbs:[Step-100]");

  vector<vector<Point>> AOxy (T);

  for (int i=0; i < num_objs; i++) {
    Point point = gen_random_point(height,width);
    AOxy[0].push_back(point);
  }
  vector<Point> Cur_Oxy = AOxy[0];
  double L1 = likelihood(image,target,Cur_Oxy,num_objs);

  for (int t=1; t<T; t++) {
    printf("\b\b\b\b%03u]", t);
    cout.flush();
    for (int i=0; i<num_objs; i++) {
      Point Oxy = Cur_Oxy[i];
      for (int j=0; j<K_MAX; j++) {
        Point Dxy = Oxy;
        Dxy.x += round(random_normal()*20);
        Dxy.y += round(random_normal()*20);
        Dxy = clamp(Dxy,height,width);
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

    }
      // show image
      imtemp = image;
      for (int n=0; n < num_objs; n++) {
        imtemp.draw_circle(Cur_Oxy[n].x, Cur_Oxy[n].y, 9, white, 0.9f, 1);
      }
      imtemp.display(main_disp);
  }
  printf("\nDone Gibbs");
  return Cur_Oxy;
}

int main(int argc, char *argv[]) {
  cimg_usage("command line arguments");
  const char *filename = cimg_option("-f",(char*)0,"Input image filename");
  init_k   = cimg_option("-n", 19,"Number of initial targets");
 
  Img image_load(filename), image, imtemp;
  Img target_load("images/target.bmp"), target;
  image = image_load.channel(0);
  target = target_load.channel(0);

  // normalize our 256 level grayscale to 0-1 float
  image /= 256.0;
  target /= 256.0;

  boost::math::poisson_distribution<> pd(lambda);

  height = image.height();
  width = image.width();

  main_disp = CImgDisplay(width,height,"Main",0);
  black = Img(width,height,1,1,0);

  printf("Image is %d x %d\n", width, height);

  vector<int> num_objs (sampling_steps);
  num_objs[0] = init_k;

  vector<double> obj_fn (sampling_steps);

  // init rng
  boost::mt19937 rng;
  boost::uniform_int<> jump(-1,1);

  main_disp.set_normalization(1);

  imtemp = image;
  vector<vector<Point>> Oxy (sampling_steps);
  for (int i=0; i < num_objs[0]; i++) {
    Point point = gen_random_point(height,width);
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
    if (a==-1 && num_objs[i-1] > 1) {
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
    printf("\n\t\tOBJ_FN:[%.5e]--AcceptRate:[%2.2f]",obj_fn[i], pa_jump);
    double u0 = random_uniform();
    if (pa_jump > u0) {
      printf("--Accept %2.2f > %2.2f", pa_jump, u0);
      // add video frame ?
    } else {
      printf("--Reject %2.2f <= %2.2f", pa_jump, u0);
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
  vector<vector<Point>> BI_AOxy, K_BI_AOxy;
  vector<int> bi_num_obj;
  int num_obj_sum = 0, final_num_obj, nsteps=0;
  for(int i=M_BURN_IN-1; i<sampling_steps; i+=STEP_BURN_IN) {
    BI_AOxy.push_back(Oxy[i]);
    bi_num_obj.push_back(num_objs[i]);
    num_obj_sum += num_objs[i];
    nsteps++;
  } 
  // step 4 - mean estimate of object number k*
  final_num_obj = round((float)num_obj_sum / nsteps);
  printf("Number of objects: %02u\n", final_num_obj);

  for (int i=0; i<BI_AOxy.size(); i++) {
    if (bi_num_obj[i] == final_num_obj)
      K_BI_AOxy.push_back(BI_AOxy[i]);
  }
  int num_sp = K_BI_AOxy.size();
  // reorder_samples
  // compute mean of samples (and round), final result
  // print final result
  // render final result
  // plot objects vs iteration
  // plot objective function vs iteration
  // render result w/best objective function
  printf("\nProgram Exit\n");
  return 0;
}
