#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
Mat affine(cv::Mat img,double tx, double ty){
    // get height and width
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    // other parameters
    int x_before, y_before;

    Mat out = cv::Mat::zeros(height, width, CV_8UC3);

    // Affine transformation
    for (int y = 0; y < height; y++){
        // get original position y
        y_before = (int)(y+ty);

        if ((y_before < 0) || (y_before >= height)){
            continue;
        }

        for (int x = 0; x < width; x++){
            // get original position x
            x_before = (int)(x+tx);

            if ((x_before < 0) || (x_before >= width)){
                continue;
            }

            // assign pixel to new position
            for (int c = 0; c < channel; c++){
                out.at<cv::Vec3b>(y, x)[c] = img.at<cv::Vec3b>(y_before, x_before)[c];
            }
        }
    }

    return out;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);

    // affine
    cv::Mat out = affine(img, -30, 30);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_28.jpg", out);

    return 0;
}