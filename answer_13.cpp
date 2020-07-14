//
// Created by YunShao on 2020/7/14.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
using namespace cv;
//使用网格内像素的最大值和最小值的差值对网格内像素重新赋值。通常用于边缘检测
Mat max_min_filter(cv::Mat img, int kernel_size){
    int height = img.rows;
    int width = img.cols;

    // prepare output
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

    int pad = floor(kernel_size / 2);
    double vmax = 0, vmin = 999, v = 0;

    // filtering
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            vmax = 0;
            vmin = 999;
            for(int dy=-pad;dy<pad+1;dy++){
                for(int dx=-pad;dx<pad+1;dx++){
                    if(((y+dy)>=0 && (x+dx)>=0) && ((y+dy)<height && (x+dx)<width) ){
                        v = (double)img.at<uchar>(y+dy,x+dx);
                        if(v>vmax){
                            vmax =v;
                        }
                        if(v<vmin){
                            vmin = v;
                        }
                    }
                }
            }
            out.at<uchar>(y, x) = (uchar)(vmax - vmin);
        }
    }
    return out;
}
int main(int argc, const char* argv[]){
    // read image
    cv::Mat gray = cv::imread("../imori.jpg", cv::IMREAD_GRAYSCALE);

    // max min filter
    cv::Mat out = max_min_filter(gray, 3);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_13.jpg", out);

    return 0;
}

