//
//  main.cpp
//  Computer Vision - Project 1
//
//  Created by Hui Hu on 1/10/23.
//


#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "filter.hpp"

using namespace cv;
using namespace std;


 int mainForImage() {
     // read the image
     Mat src = imread(getAbsolutePath("/Project1/Resources/0_image.jpeg"));
     Mat dst = src.clone();
     Mat dst0 = src.clone();
     Mat sx, sy;

     string status = "Origin";
     double alpha = 1;   // Contrast control   (1.0 - 10.0)
     double beta = 0;    // Brightness control (-200 - +200)

     // Enter a loop
     while (true) {
         imshow("Image", dst);
         char key = (char) cv::waitKey(0); // explicit cast

         switch(key){
            // Modify Contrast
             case 0:
                 alpha = min(alpha + 0.5, 10.0);
                 break;
             case 1:
                 alpha = max(alpha - 0.5, 0.0);
                 break;
            // Modify Brightness
             case 2:
                 beta = max(beta - 10, -200.0);
                 break;
             case 3:
                 beta = min(beta + 10, 200.0);
                 break;
            // Save an image
             case 's':
                 saveImage(dst, status);
                 break;
            // Grayscale (cvtColor)
             case 'g':
                 cvtColor(src, dst0, COLOR_RGBA2GRAY);
                 status = "Gray";
                 break;
            // Grayscale (alternative)
             case 'h':
                 greyscale(src, dst0);
                 status = "Gray2";
                 break;
            // Blur
             case 'b':
                  blur5x5(src, dst0);
                 status = "Blur";
                  break;
            // Sobel X
             case 'x':
                 sobelX3x3(src, sx);
                 convertScaleAbs(sx, dst0);
                 status = "SobelX";
                 break;
            // Sobel Y
             case 'y':
                 sobelY3x3(src, sy);
                 convertScaleAbs(sy, dst0);
                 status = "SobelY";
                 break;
            // Magnitude
             case 'm':
                 sobelX3x3(src, sx);
                 sobelY3x3(src, sy);
                 magnitude(sx, sy, dst0);
                 status = "Magnitude";
                 break;
            // Blur and Quantize
             case 'i':
                 blurQuantize(src, dst0);
                 status = "BlurQuantize";
                 break;
            // Cartoon
             case 'c':
                 cartoon(src, dst0);
                 status = "Cartoon";
                 break;
            // Automatic Brightness and Contrast
             case 'a':
                 automaticBC(src, dst0, 1);
                 status = "AutoBC";
                 break;
            // Remove all filters
             case 32:
                 dst0 = src.clone();
                 alpha = 1;
                 beta = 0;
                 status = "Origin";
             default:
                 break;
         }
         
         convertScaleAbs(dst0, dst, alpha=alpha, beta=beta);
         
         // exit if 'q' key or 'esc' key was pressed.
         if (key == 'q' || key == 27) break;
     }
     return 0;
 }


