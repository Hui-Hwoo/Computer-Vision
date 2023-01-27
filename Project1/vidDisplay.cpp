//
//  vidDisplay.cpp
//  Computer Vision - Project 1
//
//  Created by Hui Hu on 1/24/23.
//

#include "filter.hpp"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    VideoCapture *capdev;

    // open the video device
    capdev = new VideoCapture(0);
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    Size refS((int) capdev->get(CAP_PROP_FRAME_WIDTH ),
                  (int) capdev->get(CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    namedWindow("Video", 1); // identifies a window
    Mat frame;
    Mat dst, dst0;
    Mat sx, sy;
    
    string status = "Origin";
    double alpha = 1;   // Contrast control   (1.0 - 10.0)
    double beta = 0;    // Brightness control (-200 - +200)

    char keyValue = -1;
    for(;;) {
        // get a new frame from the camera, treat as a stream
        *capdev >> frame;
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        
        dst0 = frame.clone();
        
        switch(keyValue){
           // Modify Contrast
            case 0:
                alpha = min(alpha + 0.5, 10.0);
                keyValue = -1;
                break;
            case 1:
                alpha = max(alpha - 0.5, 0.0);
                keyValue = -1;
                break;
           // Modify Brightness
            case 2:
                beta = max(beta - 10, -200.0);
                keyValue = -1;
                break;
            case 3:
                beta = min(beta + 10, 200.0);
                keyValue = -1;
                break;
           // Save an image
            case 's':
                saveImage(dst, status);
                keyValue = -1;
                break;
           // Grayscale (cvtColor)
            case 'g':
                cvtColor(frame, dst0, COLOR_RGBA2GRAY);
                status = "Gray";
                break;
           // Grayscale (alternative)
            case 'h':
                greyscale(frame, dst0);
                status = "Gray2";
                break;
           // Blur
            case 'b':
                 blur5x5(frame, dst0);
                status = "Blur";
                 break;
           // Sobel X
            case 'x':
                sobelX3x3(frame, sx);
                convertScaleAbs(sx, dst0);
                status = "SobelX";
                break;
           // Sobel Y
            case 'y':
                sobelY3x3(frame, sy);
                convertScaleAbs(sy, dst0);
                status = "SobelY";
                break;
           // Magnitude
            case 'm':
                sobelX3x3(frame, sx);
                sobelY3x3(frame, sy);
                magnitude(sx, sy, dst0);
                status = "Magnitude";
                break;
           // Blur and Quantize
            case 'i':
                blurQuantize(frame, dst0);
                status = "BlurQuantize";
                break;
           // Cartoon
            case 'c':
                cartoon(frame, dst0);
                status = "Cartoon";
                break;
           // Automatic Brightness and Contrast
            case 'a':
                automaticBC(frame, dst0, 1);
                status = "AutoBC";
                break;
           // Remove all filters
            case 32:
                alpha = 1;
                beta = 0;
                status = "Origin";
                keyValue = -1;
            default:
                break;
        }
        
        convertScaleAbs(dst0, dst, alpha=alpha, beta=beta);

        imshow("Video", dst);


        // see if there is a waiting keystroke
        char key = waitKey(1);
        
        // exit if 'q' key or 'esc' key was pressed.
        if (key == 'q' || key == 27) break;
        
        if(key >= 0){
            if (key == 0 || key == 1 || key == 2 || key == 3 || key == 32 || key == 's' || key == 'g' || key == 'h' || key == 'b' || key == 'x' || key == 'y' || key == 'm' || key == 'i' || key == 'c' || key == 'a') {
                keyValue = key;
            }
        }
        
    }

    delete capdev;
    return(0);
}
