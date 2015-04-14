#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>

// image processing & display
#include "CImg.h"

using namespace cimg_library;

struct Point {
  int x,y;
};

int main(int argc, char *argv[]) {
  cimg_usage("command line arguments");
  const char *filename = cimg_option("-o",(char*)0,"Output .bmp filename, if not present: displays image instead");
  const int   width    = cimg_option("-r",128,"Rows of output image");
  const int   height   = cimg_option("-c",128,"Columns of output image");
  const int   discs    = cimg_option("-n", 16,"Number of targets in image");

  CImg<unsigned char> image(width,height,1,1);

  // init rng
  boost::mt19937 rng;
  boost::uniform_int<> rw(0,width-1);
  boost::uniform_int<> rh(0,height-1);
  boost::uniform_int<> noise(0,10);

  cimg_forXY(image,x,y) {
    image(x,y) = noise(rng) + 128;
  }

  for(float i=0; i<discs-1; i++) {
    Point p = Point{rw(rng), rh(rng)};
    cimg_forXY(image,x,y) {
      float radius = (x-p.x)*(x-p.x)+(y-p.y)*(y-p.y);
      if (radius < 100)
        image(x,y) = noise(rng) + 64;
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
