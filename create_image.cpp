#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

// image processing & display
#include "CImg.h"

using namespace cimg_library;

struct Point {
  int x,y;
};

int clamp(int x, int min, int max) {
  return (x<min) ? min : (x>max) ? max : x;
}

int main(int argc, char *argv[]) {
  cimg_usage("command line arguments");
  const char *filename = cimg_option("-o",(char*)0,"Output .bmp filename, if not present: displays image instead");
  const int   width    = cimg_option("-r",128,"Rows of output image");
  const int   height   = cimg_option("-c",128,"Columns of output image");
  const int   discs    = cimg_option("-n", 16,"Number of targets in image");

  CImg<unsigned char> image(width,height,1,1);

  // init rng
  boost::mt19937 rng(time(0));
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
    rw(rng, boost::uniform_int<>(0,width-1));
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
    rh(rng, boost::uniform_int<>(0,height-1));
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
    noise(rng, boost::normal_distribution<>(0.0,1.0));

  cimg_forXY(image,x,y) {
    image(x,y) = clamp(int(noise()*10.24 + 128),0,255);
  }

  for(float i=0; i<discs-1; i++) {
    Point p = Point{rw(), rh()};
    cimg_forXY(image,x,y) {
      float radius = (x-p.x)*(x-p.x)+(y-p.y)*(y-p.y);
      if (radius < 100)
        image(x,y) = clamp(int(noise()*10.24 + 64),0,255);
    }
  }

  if(filename) {
    image.save(filename);
  } else {
    CImgDisplay maindisp(width*2,height*2,"",0,false,false);
    image.display(maindisp,false,0);
  }
  return 0;
}
