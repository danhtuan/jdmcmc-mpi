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

/*
Mxy_videoseg gibss_sampling(img, tg, M) {
  printf("\n\t\tGibss:[Step/Discs-100/25]");
  int T = 50; // number of steps
  int M_BURN_IN = 50;
  int N_BURN_IN = 2;
  int K_MAX = 50;
  [rows cols] = size(img);
  %1. Initialize {zi: i = 1, ..., M}
  AOxy = zeros(M, 2, T);
  AOxy(:,:,1) = [randi(rows, M, 1) randi(cols, M, 1)];%All object position
  Cur_Oxy = AOxy(:,:,1);
  showImg = drawcircle(img, Cur_Oxy, M);
  figure(1); imshow(showImg);
  L1 = likelihood(img, tg, Cur_Oxy, M);

  Imframe(1:rows,1:cols,1)=showImg; 
  Imframe(1:rows,1:cols,2)=showImg; 
  Imframe(1:rows,1:cols,3)=showImg;
  videoseg(1) = im2frame(Imframe);          % make the first frame
  for t = 2:T
      for i = 1:M        
          fprintf('\b\b\b\b\b\b\b');            
          fprintf('%03u/%02u]', t, i);
          Oxy = Cur_Oxy(2*i-1:2*i);%init position of ith object
          for j = 1:K_MAX            
              %Sampling ith variable            
              Dxy = Oxy + round(randn(1,2)*20);
              Dxy=clip(Dxy,1,rows);% make sure the position in the image
              New_Cur_Oxy = Cur_Oxy;
              New_Cur_Oxy(2*i-1:2*i) = Dxy;
              L2=likelihood(img,tg,New_Cur_Oxy,M);% evaluate the likelihood
              v=min(1,L2/L1);                     % compute the acceptance ratio
              u=rand;                             % draw a sample uniformly in [0 1]
              if v>u                
                  Oxy = Dxy;% accept the move                        
                  Cur_Oxy = New_Cur_Oxy;
                  L1 = L2;                
  %                 showImg = drawcircle(img, Cur_Oxy, M);
  %                 figure(1); imshow(showImg);
              else                       
              end            
          end
          AOxy(:,:, t) = Cur_Oxy;       
          showImg = drawcircle(img, Cur_Oxy, M);
          figure(1); imshow(showImg);
          Imframe(1:rows,1:cols,1)=showImg; 
          Imframe(1:rows,1:cols,2)=showImg; 
          Imframe(1:rows,1:cols,3)=showImg;
          videoseg((t-2)*M + i + 1) = im2frame(Imframe);          
      end
  end
  fprintf('...Burning...');
  %Burn-in
  % S = AOxy(:,:, M_BURN_IN+1:N_BURN_IN:T);%do burn-in, drop fist M samples, keep N-steps samples
  % OS = reorder_samples(S);
  % Mxy = round(mean(OS, 3));
  Mxy = Cur_Oxy;
  showImg = drawcircle(img, Mxy, M);figure(1);imshow(showImg);
  Imframe(1:rows,1:cols,1)=showImg; 
  Imframe(1:rows,1:cols,2)=showImg; 
  Imframe(1:rows,1:cols,3)=showImg;
  videoseg((T-1) * M + 2) = im2frame(Imframe);          
  fprintf('Done Gibss');
}

*/


double likelihood(const CImg<unsigned char> &image, const CImg<unsigned char> &target, vector<Point> Uxy, int K) {
  int x = image.height();
  int y = image.width();

  CImg<unsigned char> It(128,128); // = maketarget(t,height,width,Uxy,K);
  It.fill(0);
  
  for (int i=0; i<K; i++) {
    It(Uxy[i].x, Uxy[i].y) = 1;
  }
  //It.display("It");

  CImg<unsigned char> Ie=-It.get_convolve(target, 0, false).get_threshold(128).get_normalize(129,193);
  (image,Ie).display();
  //Ie.display("Ie-conv");

  double mse = image.MSE(Ie);

  printf("MSE %.3f\n", mse);
  
  return exp(-0.5*mse);
}

int main() {
  CImg<unsigned char> image("discs20.bmp"), imorig;
  CImg<unsigned char> target("target.bmp");
  imorig = image;
  unsigned char t[19][19];
  boost::math::poisson_distribution<> pd(lambda);
  cimg_forXY(target,x,y) {
    t[x][y] = target(x,y)/255;
  }

  /* 
  for(int i=0; i<19; i++) {
    for(int j=0; j<19; j++)
      printf("%u\t", t[i][j]);
    printf("\n");
  }
  */

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
    //printf("%d %d\n", Oxy[i].x, Oxy[i].y);
    float white[3] = {255,255,255};
    image.draw_circle(Oxy[i].x, Oxy[i].y, 9, white, 0.9f, 1);
  } 
  image.display(main_disp, 0);

  // showImg = drawcircle(img, points, num_objs)
  // show figure
  obj_fn[0] = likelihood(imorig, target, Oxy, num_objs[0]) * pdf(pd,num_objs[0]);

  // Imframe stuff = showImg (r,g,b)

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
    printf("--Discs:[%02u]", num_objs[i]);
    // gibbs sampling
    // accept/reject
    //obj_fn[i] = likelihood() * poisspdf();
    //pa_jump = min(obj_fn[i]/obj_fn[i-1]);
    pa_jump = obj_fn[i]/obj_fn[i-1];
  }

  return 0;
}
