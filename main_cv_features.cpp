#include <stdint.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


using std::cout;
using std::endl;




int main(  )
{
    string nom_objeto="../Data/ima1.png";
    string nom_secene="../Data/ima21.png";

     Mat fto_objeto = imread(nom_objeto , IMREAD_GRAYSCALE );

     Mat fto_scene = imread(nom_secene , IMREAD_GRAYSCALE );


    if ( (fto_objeto.empty()) || (fto_scene.empty()) )
    {
        cout << "Could not open or find the image!\n" << endl;
         return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detecto
   //Ptr<SIFT> detector_sift;
   //Ptr<BRISK> detector_brisk;
   //Ptr<FREAK> detector_freak;

   int nfeatures = 0;
   int n_OctaveLayers = 3;
   double contrastThreshold = 0.04;
   double edgeThreshold = 10;
   double sigma = 1.6;
   Ptr<SIFT> feat_sift=SIFT::create(nfeatures,n_OctaveLayers,contrastThreshold,edgeThreshold,sigma);

   int thresh=30;
   int octavesb=3;
   float patternScaleb=1.0f;
   Ptr<BRISK> feat_brisk=BRISK::create(thresh,octavesb,patternScaleb);

   bool orientationNormalized = true;
   bool scaleNormalized = true;
   float patternScalef = 22.0f;
   int nOctavesf = 4;
  //Ptr<FREAK> k_freak;
  Ptr<FREAK> feat_freak=FREAK::create(orientationNormalized,scaleNormalized,patternScalef,nOctavesf);

                               // con 50 en hessina me ngenreo 773 keypoints
   double hessianThreshold=50; // con 100 en hessina me ngenreo 400 keypoints
                               // con 500 en hessina me ngenreo 159 keypoints
   int nOctaves = 4;
   int nOctaveLayers = 3;
   bool extended = false;
   bool upright = false;
   Ptr<SURF> feat_surf = SURF::create( hessianThreshold,nOctaves,nOctaveLayers,extended,upright );
   //int minHessian = 400;
  // Ptr<SURF> detector_surf = SURF::create( minHessian );

   std::vector<KeyPoint> keypoints_surf_objeto,keypoints_surf_scene;
   std::vector<KeyPoint> keypoints_sif_objeto,keypoints_sif_scene;
   std::vector<KeyPoint> keypoints_brisk;

   Mat Descrip_surf_surf_obeto;
   feat_surf->detect(fto_objeto,keypoints_surf_objeto );
   feat_surf->compute(fto_objeto,keypoints_surf_objeto,Descrip_surf_surf_obeto);

   Mat Descrip_surf_surf_scene;
   feat_surf->detect(fto_scene,keypoints_surf_scene );
   feat_surf->compute(fto_scene,keypoints_surf_scene,Descrip_surf_surf_scene);



   Mat Descrip_sift_surf;
   feat_sift->detect(fto_objeto,keypoints_sif_objeto);
   feat_sift->detect(fto_scene,keypoints_sif_scene);



   Mat Descrip_surf_sift_objeto;
   Mat Descrip_surf_sift_scene;

   feat_sift->compute(fto_objeto,keypoints_surf_objeto,Descrip_surf_sift_objeto);
   feat_sift->compute(fto_scene,keypoints_surf_scene,Descrip_surf_sift_scene);

   //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
   std::vector< std::vector<DMatch> > matches_surf_sift;
   matcher->knnMatch( Descrip_surf_sift_objeto, Descrip_surf_sift_scene, matches_surf_sift, 2 );


    //-- Draw keypoints
    Mat img_keypoints;
    //drawKeypoints( foto, keypoints_surf, img_keypoints );
    //-- Show detected (drawn) keypoints
    //imshow("SURF Keypoints", img_keypoints );
   // waitKey();
    return 0;
}

