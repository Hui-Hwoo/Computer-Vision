//
//  filter.cpp
//  Test
//
//  Created by Hui Hu on 1/24/23.
//

// basic
#include "filter.hpp"
#include <iostream>
#include <cstdlib>
#include <algorithm>

// opencv
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// string
#include <filesystem>
#include <string>
#include <sstream>

// time
#include <chrono>

using namespace std;
using namespace cv;
using std::filesystem::current_path;

// helper function, takes the relative path and returns the absolute path
string getAbsolutePath(string relativePath){
    // get current directory path
    filesystem::path directoryPath = current_path();
    string stringpath = directoryPath.generic_string();

    // combine current directory path with relative path
    stringpath.append(relativePath);
    return stringpath;
}


// rename image and save it to Resources folder
int saveImage(Mat originalImage, string prefix, string format){
    // get seconds since epoch
    const auto current_time = std::chrono::system_clock::now();
    long seconds = chrono::duration_cast<std::chrono::seconds>(current_time.time_since_epoch()).count();
    
    // convert long seconds to string timeStamp
    stringstream stream;
    stream << seconds;
    string timeStamp = stream.str();
    
    string imageName = "/Project1/Resources/" + prefix + "_" + timeStamp + "." + format;
    imwrite(getAbsolutePath(imageName), originalImage.clone());
    cout << imageName << endl;
    return 0; 
}

// customized function to convert an image to gray scale
int greyscale(Mat &src, Mat &dst){
    dst = Mat::zeros(src.rows, src.cols, src.type());
        
    Mat_<Vec3b> dstVec = dst;
    Mat_<Vec3b> srcVec = src;
    for( int i = 0; i < src.rows; ++i){
        for( int j = 0; j < src.cols; ++j ){
            const int value = (srcVec(i, j)[0] + srcVec(i, j)[1] + srcVec(i, j)[2]) / 3;
            dstVec(i, j)[0] = value;
            dstVec(i, j)[1] = value;
            dstVec(i, j)[2] = value;
        }
    }
    
    return 0;
}

// Implement a 5x5 Gaussian filter as separable 1x5 filters ([1 2 4 2 1] vertical and horizontal)
int blur5x5(Mat &src, Mat &dst){
    dst = Mat::zeros(src.rows, src.cols, src.type());
    
    Mat_<Vec3b> dstVec = dst;
    Mat_<Vec3w> tmpVec = Mat::zeros(dst.rows, dst.cols, CV_16UC3); // temporary vector, increases precision
    Mat_<Vec3b> srcVec = src;

    const int filter[5] = {1,2,4,2,1};
    
    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;
    int d,r,c,k;                // loop variables
    unsigned short count;

    // Non-separable filter (time-consuming)
//    const int filter2D[5][5] = {{1,2,4,2,1}, {2,4,8,4,2}, {4,8,16,8,4}, {2,4,8,4,2}, {1,2,4,2,1}};
//    int i, j;
//    for (d=0; d<channels; d++) {
//        for (r=2; r<rows-2; r++) {
//            for (c=2; c<cols-2; c++) {
//                for (i=r-2; i<=r+2; i++) {
//                    for (j=c-2; j<=c+2; j++) {
//                        tmpVec(r,c)[d] += filter2D[i-r+2][j-c+2]*srcVec(i,j)[d];
//                    }
//                }
//                dstVec(r,c)[d] = tmpVec(r,c)[d] / 100;
//            }
//        }
//    }

    // Separable filter (time-efficient)
    // Vertical direction
    for (d=0; d<channels; d++) {
        for (r=2; r<rows-2; r++) {
            for (c=0; c<cols; c++) {
                for (k=r-2; k<=r+2; k++) {
                    tmpVec(r,c)[d] += filter[k-r+2]*srcVec(k,c)[d];
                }
            }
        }
    }

    // Horizontal direction
    for (d=0; d<channels; d++) {
        for (r=2; r<rows-2; r++) {
            for (c=2; c<cols-2; c++) {
                count = 0;
                for (k=c-2; k<=c+2; k++) {
                    count += filter[k-c+2]*tmpVec(r,k)[d];
                }
                dstVec(r,c)[d] = count / 100;
            }
        }
    }
    
    // Deal with the edges
    for (d=0; d<channels; d++) {
        for (r=0; r<rows; r++) {
            for (c=0 ; c<cols; c++) {
                if(r < 2 || r >= rows - 2 || c < 2 || c >= cols - 2){
                    dstVec(r,c)[d] = dstVec(min(max(r,2), rows-3),min(max(c,2), cols-3))[d];
                }
            }
        }
    }

    return 0;
}

// Helper function, takes the images and separable filters, made the modification
// filterV: vertical filter
// filterH: horizontal filter
void filterHelper(cv::Mat &src, cv::Mat &dst, const int* filterV, const int* filterH){
    
    Mat_<Vec3s> dstVec = dst;
    const Mat_<Vec3b> srcVec = src;
    Mat_<Vec3s> tmpVec = Mat::zeros(src.rows, src.cols, CV_16SC3);

    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;
    int d,r,c,k;

    // Vertical direction
    for (d=0; d<channels; d++) {
        for (r=1; r<rows-1; r++) {
            for (c=0; c<cols; c++) {
                for (k=r-1; k<=r+1; k++) {
                    tmpVec(r,c)[d] += filterV[k-r+1] * srcVec(k,c)[d];
                }
            }
        }
    }

    // Horizontal direction
    for (d=0; d<channels; d++) {
        for (r=1; r<rows-1; r++) {
            for (c=1; c<cols-1; c++) {
                for (k=c-1; k<=c+1; k++) {
                    dstVec(r,c)[d] += filterH[k-c+1] * tmpVec(r,k)[d];
                }
                dstVec(r,c)[d] = dstVec(r,c)[d];
            }
        }
    }

    // Deal with the edges
    for (d=0; d<channels; d++) {
        for (r=0; r<rows; r++) {
            for (c=0 ; c<cols; c++) {
                if(r < 1 || r >= rows - 1 || c < 1 || c >= cols - 1){
                    dstVec(r,c)[d] = dstVec(min(max(r,1), rows - 2),min(max(c,1), cols - 2))[d];
                }
            }
        }
    }
}

// Implement a 3x3 Sobel filter, either horizontal (X) or vertical (Y)
int sobelX3x3(cv::Mat &src, cv::Mat &dst){
    dst = Mat::zeros(src.rows, src.cols, CV_16SC3);
    
    /*
           | +1  0  -1 |   |1|
     G_x = | +2  0  -2 | = |2| * [+1 0 -1]
           | +1  0  -1 |   |1|
     */
    
    const int filterV[3] = {1,2,1};
    const int filterH[3] = {1,0,-1};
    filterHelper(src, dst, filterV, filterH);
    
    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst){
    dst = Mat::zeros(src.rows, src.cols, CV_16SC3);
    
    /*
           | +1  +2  +1 |   |+1|
     G_y = |  0   0   0 | = | 0| * [1 2 1]
           | -1  -2  -1 |   |-1|
     */
    
    const int filterV[3] = {1,0,-1};
    const int filterH[3] = {1,2,1};
    filterHelper(src, dst, filterV, filterH);

    return 0;
}

// Implement a function that generates a gradient magnitude image using Euclidean distance for magnitude
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
    Mat_<Vec3s> sxVec = sx;
    Mat_<Vec3s> syVec = sy;
    Mat_<Vec3s> tmpVec = Mat::zeros(sx.rows, sx.cols, CV_16SC3);
    
    int channels = sx.channels();
    int rows = sx.rows;
    int cols = sx.cols;
    int d,r,c;


    for (d=0; d<channels; d++) {
        for (r=0; r<rows; r++) {
            for (c=0; c<cols; c++) {
                tmpVec(r, c)[d] = sqrt(sxVec(r, c)[d] * sxVec(r, c)[d] + syVec(r, c)[d] * syVec(r, c)[d]);
            }
        }
    }
    
    convertScaleAbs(tmpVec, dst);

    return 0;
}

// Implement a function that blurs and quantizes a color image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels){
    // Blur the image
    blur5x5(src, dst);
    
    // Quantize the image
    Mat_<Vec3b> dstVec = dst;
    Mat_<Vec3b> srcVec = src;
    
    int b = 255 / levels;
    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;
    int d,r,c;
    
    for (d=0; d<channels; d++) {
        for (r=0; r<rows; r++) {
            for (c=0; c<cols; c++) {
                dstVec(r, c)[d] = dstVec(r, c)[d] / b;
                dstVec(r, c)[d] = dstVec(r, c)[d] * b;
            }
        }
    }

    return 0;
}

// Implement a live video cartoonization function using the gradient magnitude and blur/quantize filters
int cartoon(cv::Mat &src, cv::Mat&dst, int levels, int magThreshold){
    // Calculate the gradient magnitude
    Mat sx, sy, mag;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    magnitude(sx, sy, mag);
    
    // Blur and quantize the image
    blurQuantize(src, dst, levels);
    
    // Black pixels with a gradient magnitude larger than a threshold
    Mat_<Vec3b> dstVec = dst;
    Mat_<Vec3b> magVec = mag;
    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;
    int d,r,c;
//    int maxi = 0;
    
    for (r=0; r<rows; r++) {
        for (c=0; c<cols; c++) {
            for(d=0; d<channels; d++){
                if(magVec(r, c)[d] > magThreshold){
                    dstVec(r, c)[d] = 0;
                }
//                maxi = 0;
//                for(d=0; d<channels; d++){
//                    if(maxi < magVec(r, c)[d]){
//                        maxi = magVec(r, c)[d];
//                    }
//                }
//                if(maxi > magThreshold){
//                    for(d=0; d<channels; d++){
//                        dstVec(r, c)[d] = 0;
//                    }
//                }
            }
        }
    }
    
    return 0;
}

// Automatic brightness and contrast optimization with optional histogram clipping
int automaticBC(cv::Mat &src, cv::Mat&dst, float clip_hist_percent){
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    
    // Calculate grayscale histogram
    Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange[] = { range };
    calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, histRange, true, false);

    // Calculate cumulative distribution from the histogram
    vector<float> accumulator = {};
    accumulator.push_back(hist.at<float>(0));
    for (int i = 1; i < histSize; i++) {
        accumulator.push_back(accumulator[i-1] + hist.at<float>(i));
    }
    
    // Locate points to clip
    float maximum = accumulator[accumulator.size()-1];
    clip_hist_percent *= (maximum/100.0);
    clip_hist_percent /= 2.0;

    // Locate left cut
    int minimum_gray = 0;
    while (accumulator[minimum_gray] < clip_hist_percent) {
        minimum_gray++;
    }

    // Locate right cut
    int maximum_gray = histSize - 1;
    while (accumulator[maximum_gray] >= (maximum - clip_hist_percent)) {
        maximum_gray--;
    }
    
    // Calculate alpha and beta values
    float alpha = 255.0 / (maximum_gray - minimum_gray);
    float beta = -minimum_gray * alpha;

    convertScaleAbs(src, dst, alpha, beta);
    
    return 0;
}
