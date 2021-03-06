#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

// C headers (ew)
#include <mpi.h>
#include <time.h>

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

int init_k;
int width;
int height;

// will be initialized in main with time+world_rank*n
boost::mt19937 gen;

CImgDisplay main_disp;
double white[1] = {1.0};
int world_size, world_rank;
char processor_name[MPI_MAX_PROCESSOR_NAME];
bool show_disp = true;

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

  double sse = Ie.MSE(black)*width*height; // note: gibbs is sensitive to this
  double like = exp(-0.5*sse);
  return like;
}

Point gen_random_point(int width, int height) {
  using namespace boost;

  variate_generator<mt19937&, uniform_int<> >
    randcol(gen, uniform_int<>(0,width-1));
  variate_generator<mt19937&, uniform_int<> >
    randrow(gen, uniform_int<>(0,height-1));

  return Point{randcol(), randrow()};
}

int clamp(int x, int min, int max) {
  return (x<min) ? min : (x>max) ? max : x;
}

double clamp(double x, double min, double max) {
  return (x<min) ? min : (x>max) ? max : x;
}

Point clamp(Point p, int w, int h) {
  if (p.x > w-1)
    p.x = (w-1) - (p.x-(w-1)); 
  else if (p.x < 0)
    p.x = -p.x;
  if (p.y > h-1)
    p.y = (h-1) - (p.y-(h-1)); 
  else if (p.y < 0)
    p.y = -p.y;
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
  const int max_iterations, int diffusion) {
  int T = max_iterations;
  //if (world_rank == 0) printf("\n\t\tGibbs:[Step-100]");

  vector<Point> Cur_Oxy = Oxy;
  double L1 = likelihood(image,target,Cur_Oxy,num_objs);

  for (int t=0; t<T; t++) {
    for (int i=0; i<num_objs; i++) {
      Point Dxy = Cur_Oxy[i];
      Dxy.x += round(random_normal()*diffusion);
      Dxy.y += round(random_normal()*diffusion);
      Dxy = clamp(Dxy,width,height);
      vector<Point> New_Cur_Oxy = Cur_Oxy;
      New_Cur_Oxy[i] = Dxy;
      double L2 = likelihood(image,target,New_Cur_Oxy,num_objs);
      double v = min(1.0,L2/L1);
      double u = random_uniform();
      if (v > u) {
        Cur_Oxy = New_Cur_Oxy;
        L1 = L2;
      }
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

// source:
// http://www.guyrutenberg.com/2007/09/22/profiling-code-using-clock_gettime/
timespec diff(timespec ts_start, timespec ts_end) {
  timespec d;
  if ((ts_end.tv_nsec-ts_start.tv_nsec) < 0) {
    // if we didn't check this, we'd sometimes get negative nanoseconds
    // so we borrow a 1 from the seconds and add a billion to the ns
    d.tv_sec = ts_end.tv_sec - ts_start.tv_sec - 1;
    d.tv_nsec= 1000000000 + ts_end.tv_nsec - ts_start.tv_nsec;
  } else {
    d.tv_sec = ts_end.tv_sec - ts_start.tv_sec;
    d.tv_nsec= ts_end.tv_nsec - ts_start.tv_nsec;
  }
  return d;
}


Img create_random_image(int width, int height, int discs, double variance) {
  Img image(width,height,1,1);

  // init rng
  FILE *fp;
  fp = fopen("/dev/urandom", "r");
  int seed;
  fread(&seed, 1, 4, fp);
  boost::mt19937 rng(seed);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
    rw(rng, boost::uniform_int<>(0,width-1));
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
    rh(rng, boost::uniform_int<>(0,height-1));
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
    noise(rng, boost::normal_distribution<>(0.0,1.0));

  cimg_forXY(image,x,y) {
    image(x,y) = clamp(noise()*variance+0.5,0.0,1.0);
  }

  for(float i=0; i<discs; i++) {
    Point p = Point{rw(), rh()};
    cimg_forXY(image,x,y) {
      float radius = (x-p.x)*(x-p.x)+(y-p.y)*(y-p.y);
      if (radius < 100)
        image(x,y) = clamp(noise()*variance+0.25,0.0,1.0);
    }
  }

  return image;
}

int main(int argc, char *argv[]) {
  // from MFG programming assignment 1
  timespec ts_start, ts_end;
  // get start time
  clock_gettime(CLOCK_REALTIME, &ts_start);

  MPI_Init(&argc, &argv);
  init_mpi();
  
  cimg_usage("command line arguments");
  init_k = cimg_option("-n", 19,"Number of initial targets");
  const int max_iterations = cimg_option("-i", 100,"Number of iterations per broadcast loop");
  const int max_gibbs_iterations = cimg_option("-g", 100,"Number of iterations per gibbs sampling loop");
  const double epsilon = cimg_option("-e", 1.0e-20,"stopping point for likelihood");
  const double variance = cimg_option("-v", 0.04,"variance for random image noise");
  show_disp = cimg_option("-d", true,"show display");
  const int r = cimg_option("-r", 128,"image height");
  const int c = cimg_option("-c", 128,"image width");
  const int tag = cimg_option("-t", 0,"run tag");

  int image_size, target_size;
 
  Img image(c,r,1,1), imtemp, target(19,19,1,1);
  if (world_rank == 0) { 
    Img target_load("images/target.bmp");
    target = target_load.channel(0);

    // normalize our 256 level grayscale to 0-1 float
    target /= 256.0;

    image = create_random_image(c,r,init_k,variance);

    image_size = image.size();
    target_size = target.size();
  }  

  MPI_Bcast(&image_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&target_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(image, image_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(target, target_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  height = image.height();
  width = image.width();
  printf("%s %d: %d x %d\n", processor_name, world_rank, width, height);

  if (world_rank == 0 && show_disp) main_disp = CImgDisplay(width,height,"Main",0);
  black = Img(width,height,1,1,0);

  if (world_rank == 0)
    printf("Image is %d x %d\n", width, height);

  int num_objs = init_k;

  double obj_fn;

  // init rng
  gen.seed(time(0)+world_rank*100); // avoid overlapping times across machines
  boost::uniform_int<> jump(-1,1);

  if (world_rank == 0 && show_disp) main_disp.set_normalization(1);

  imtemp = image;
  vector<Point> Oxy (0);

  for (int i=0; i < num_objs; i++) {
    Point point = gen_random_point(width,height);
    Oxy.push_back(point);
    if (world_rank == 0)
      imtemp.draw_circle(point.x, point.y, 9, white, 0.9f, 1);
  }

  // show figure
  if (world_rank == 0 && show_disp) imtemp.display(main_disp);
  obj_fn = likelihood(image, target, Oxy, num_objs);// * pdf(pd,num_objs);

  printf("Rank-[%02u]--Discs:[%02u]--OBJ_FN:[%.5e]\n", world_rank, num_objs, obj_fn);

  vector<Point> Best_Oxy, CurBest_Oxy;
  double best_obj = -1.0;

  for (int i=0; i<max_iterations; i++) {
    // variance ranges between 5-60, steps of 5, starting with 20 at rank 0
    Oxy = gibbs_sampling(image, target, num_objs, Oxy, max_gibbs_iterations, (world_rank+3)*5 % 60 + 5);
    obj_fn = likelihood(image, target, Oxy, num_objs);
  
    maxloc.val = obj_fn;
    maxloc.rank = world_rank;

    // everyone figures out which process has the best obj function value
    MPI_Allreduce( &maxloc, &r_maxloc, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD );

    CurBest_Oxy = Oxy;
    // the process with the best obj function value is the 4th arg here, it will be
    // the one that broadcasts its points list to the others
    MPI_Bcast( &CurBest_Oxy.front(), num_objs*2, MPI_INT, r_maxloc.rank, MPI_COMM_WORLD );


    if (world_rank == 0) {
      printf("Best: %d %.5e\n", r_maxloc.rank, r_maxloc.val);

      if (r_maxloc.val > best_obj) {
        printf("^^ New winner ^^\n", r_maxloc.rank, r_maxloc.val);
        best_obj = r_maxloc.val;
        Best_Oxy = CurBest_Oxy;

        imtemp = image;
        for (int n=0; n < num_objs; n++) {
          imtemp.draw_circle(Oxy[n].x, Oxy[n].y, 9, white, 0.9f, 1);
        }
      } 

      if (show_disp) {
        Img imtemp2 = image;
        for (int n=0; n < num_objs; n++) {
          imtemp2.draw_circle(CurBest_Oxy[n].x, CurBest_Oxy[n].y, 9, white, 0.9f, 1);
        }
        main_disp.resize(width*4,height*2);
        (imtemp,imtemp2).display(main_disp);
      }
    }

    Oxy = CurBest_Oxy;

    if (r_maxloc.val > epsilon) {
      if (world_rank == 0)
        printf("%.5e > %.5e epsilon ... breaking\n", best_obj, epsilon);
      break;
    }
  } // end global iterations

  if (world_rank == 0) {
    if (best_obj > epsilon) {
      printf("SUCCESS ");
    } else {
      printf("FAILURE "); // we hit max iterations
    }
    // get finish time and calculate running time
    clock_gettime(CLOCK_REALTIME, &ts_end);
    timespec ts_diff = diff(ts_start, ts_end);
    printf("Elapsed time: %d.%09d seconds %.5e\n", ts_diff.tv_sec, ts_diff.tv_nsec, best_obj);
  }

  if (world_rank == 0) {
    printf("Best: %.5e\n", best_obj);
    for (int i=0; i<num_objs; i++)
      printf("%d %d\n", Best_Oxy[i].x, Best_Oxy[i].y);

    printf("Showing best\n");
    if (show_disp) {
      main_disp.resize(width*2,height*2);
      imtemp.display(main_disp,0);
    }
    char outfile[256];
    snprintf(outfile, 256, "output/%d_%d_%s.bmp", tag, time(0), processor_name);
    imtemp.normalize(0,255);
    imtemp.save(outfile);
    printf("\nProgram Exit\n");
  }

  MPI_Finalize();
  return 0;
}
