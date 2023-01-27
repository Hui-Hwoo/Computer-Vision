//
//  filter.hpp
//  Test
//
//  Created by Hui Hu on 1/24/23.
//

#ifndef filter_hpp
#define filter_hpp

#include <stdio.h>
#include <string>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

// Get the absolute path of one image
string getAbsolutePath(string relativePath);

// Rename an image and save it to Resources folder
int saveImage(Mat originalImage, string prefix = "origin", string format= "jpg");

// customized function to convert an image to gray scale
int greyscale(Mat &src, Mat &dst);

// Implement a 5x5 Gaussian filter as separable 1x5 filters ([1 2 4 2 1] vertical and horizontal)
int blur5x5(cv::Mat &src, cv::Mat &dst);

// Implement a 3x3 Sobel filter, either horizontal (X) or vertical (Y)
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Implement a function that generates a gradient magnitude image using Euclidean distance for magnitude
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// Implement a function that blurs and quantizes a color image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels = 15);

// Implement a live video cartoonization function using the gradient magnitude and blur/quantize filters
int cartoon(cv::Mat &src, cv::Mat&dst, int levels = 15, int magThreshold = 30);

// Automatic brightness and contrast optimization with optional histogram clipping
int automaticBC(cv::Mat &src, cv::Mat&dst, float clip_hist_percent=1);

#endif /* filter_hpp */
