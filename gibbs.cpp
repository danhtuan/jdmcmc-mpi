#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>

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
int M_BURN_IN = 50;
int STEP_BURN_IN = 5;

int width;
int height;

// XXX: make this selectable via command line arguments ?
boost::mt19937 gen;
//boost::mt19937 gen(time(0)); // seeded version

CImgDisplay main_disp;
double white[1] = {1.0};
int world_size, world_rank;
char processor_name[MPI_MAX_PROCESSOR_NAME];

struct Point {
  int x,y;
};

typedef struct {
  double val;
  int rank;
} MAXLOC;

MAXLOC maxloc, r_maxloc;

typedef CImg<double> Img; 
Img black;

double likelihood(const Img &image,
  const Img &target, vector<Point> Uxy, int K) {

  Img Ie(width,height,1,1,0);

  for (int i=0; i < K; i++) {
    Ie.draw_image(Uxy[i].x-9, Uxy[i].y-9, target, target, 1.0);
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
  const Img &target, int num_objs, vector<Point> Oxy,
  const int max_iterations) {
  int T = max_iterations;
  //if (world_rank == 0) printf("\n\t\tGibbs:[Step-100]");

  vector<vector<Point>> AOxy (T);
  AOxy[0] = Oxy;
  vector<Point> Cur_Oxy = AOxy[0];
  double L1 = likelihood(image,target,Cur_Oxy,num_objs);

  for (int t=0; t<T; t++) {
    /*if (world_rank == 0)
      printf("\b\b\b\b%03u]", t);
    cout.flush();*/
    for (int i=0; i<num_objs; i++) {
      Point Oxy = Cur_Oxy[i];
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
      AOxy[t] = Cur_Oxy;
    }
  }
  //if (world_rank==0) printf("\nDone Gibbs\n");
  return Cur_Oxy;
}

void init_mpi() {
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Get the name of the processor
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  init_mpi();
  
  cimg_usage("command line arguments");
  const char *filename = cimg_option("-f",(char*)0,"Input image filename");
  init_k = cimg_option("-n", 19,"Number of initial targets");
  const int max_iterations = cimg_option("-i", 100,"Number of iterations per broadcast loop");
  const int max_gibbs_iterations = cimg_option("-g", 100,"Number of iterations per gibbs sampling loop");
  int image_size, target_size;
 
  Img image(128,128,1,1), imtemp, target(19,19,1,1);
  if (world_rank == 0) { 
    Img image_load(filename), imtemp;
    Img target_load("images/target.bmp");
    image = image_load.channel(0);
    target = target_load.channel(0);

    // normalize our 256 level grayscale to 0-1 float
    image /= 256.0;
    target /= 256.0;

    image_size = image.size();
    target_size = target.size();
  }  

  MPI_Bcast(&image_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&target_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(image, image_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(target, target_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  boost::math::poisson_distribution<> pd(lambda);

  height = image.height();
  width = image.width();
  printf("%s %d: %d x %d\n", processor_name, world_rank, width, height);

  if (world_rank == 0)
    main_disp = CImgDisplay(width,height,"Main",0);
  black = Img(width,height,1,1,0);

  if (world_rank == 0)
    printf("Image is %d x %d\n", width, height);

  int num_objs = init_k;

  double obj_fn;

  // init rng
  boost::mt19937 rng;
  gen.seed(time(0)+world_rank*100); // avoid overlapping times across machines
  boost::uniform_int<> jump(-1,1);

  if (world_rank == 0)
    main_disp.set_normalization(1);

  imtemp = image;
  vector<Point> Oxy (init_k);

  for (int i=0; i < num_objs; i++) {
    Point point = gen_random_point(height,width);
    Oxy.push_back(point);
    if (world_rank == 0)
      imtemp.draw_circle(point.x, point.y, 9, white, 0.9f, 1);
  }

  // show figure
  if (world_rank == 0)
    imtemp.display(main_disp);
  obj_fn = likelihood(image, target, Oxy, num_objs);// * pdf(pd,num_objs);

//  if (world_rank == 0)
    printf("Rank-[%02u]--Discs:[%02u]--OBJ_FN:[%.5e]\n", world_rank, num_objs, obj_fn);

  vector<Point> Best_Oxy, CurBest_Oxy;
  double best_obj = -1.0;
for (int i=0; i<max_iterations; i++) {
  Oxy = gibbs_sampling(image, target, num_objs, Oxy, max_gibbs_iterations);
  //obj_fn = likelihood(image, target, Oxy, num_objs) * pdf(pd,num_objs);
  obj_fn = likelihood(image, target, Oxy, num_objs);
  
  maxloc.val = obj_fn;
  maxloc.rank = world_rank;
  // everyone figures out which process has the best obj function value
  MPI_Allreduce( &maxloc, &r_maxloc, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD );

  // the process with the best obj function value is the 4th arg here, it will be
  // the one that broadcasts its points list to the others
  CurBest_Oxy = Oxy;
  MPI_Bcast( &CurBest_Oxy.front(), num_objs*2, MPI_INT, r_maxloc.rank, MPI_COMM_WORLD );
  if (obj_fn/r_maxloc.val > random_uniform()) {
    Oxy = CurBest_Oxy;
  }

  if (world_rank == 0) {
    printf("Best: %d %.5e\n", r_maxloc.rank, r_maxloc.val);
    if (r_maxloc.val > best_obj) {
      printf("^^ New winner ^^\n", r_maxloc.rank, r_maxloc.val);
      best_obj = r_maxloc.val;
      Best_Oxy = Oxy;
      // show image
      imtemp = image;
      for (int n=0; n < num_objs; n++) {
        imtemp.draw_circle(Oxy[n].x, Oxy[n].y, 9, white, 0.9f, 1);
      }
    } 
    Img imtemp2 = image;
    for (int n=0; n < num_objs; n++) {
      imtemp2.draw_circle(CurBest_Oxy[n].x, CurBest_Oxy[n].y, 9, white, 0.9f, 1);
    }
    main_disp.resize(256,128);
    (imtemp,imtemp2).display(main_disp);
  }
}

  if (world_rank == 0) {
    printf("Showing best\n");
    main_disp.resize(512,512);
    imtemp.display(main_disp,0);
  }
  // reorder_samples
  // compute mean of samples (and round), final result
  // print final result
  // render final result
  // plot objects vs iteration
  // plot objective function vs iteration
  // render result w/best objective function

  double mpi_obj_fn[36]; // XXX: hardcoded to our world max
  MPI_Gather( &obj_fn, 1, MPI_DOUBLE, mpi_obj_fn, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );

  MPI_Finalize();

  if (world_rank == 0) {
    printf("Best: %.5e\n", best_obj);
    for (int i=0; i<num_objs; i++)
      printf("%d %d\n", Best_Oxy[i].x, Best_Oxy[i].y);
    printf("\nProgram Exit\n");
  }

  return 0;
}
