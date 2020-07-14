//
// Created by YunShao on 2020/7/14.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
//差分滤波器对图像亮度急剧变化的边缘有提取效果，可以获得邻接像素的差值
using namespace cv;

Mat diff_filter(cv::Mat img, int kernel_size, bool horizontal){
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();

    // prepare output
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

    // prepare kernel
    double kernel[kernel_size][kernel_size] = {{0, -1, 0}, {0, 1, 0}, {0, 0, 0}};

    if (horizontal){
        kernel[0][1] = 0;
        kernel[1][0] = -1;
    }

    int pad = floor(kernel_size / 2);

    double v = 0;

    // filtering
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            v = 0;
            for (int dy = -pad; dy < pad + 1; dy++){
                for (int dx = -pad; dx < pad + 1; dx++){
                    if (((y + dy) >= 0) && (( x + dx) >= 0) && ((y + dy) < height) && ((x + dx) < width)){
                        v += img.at<uchar>(y+dy,x+dx)*kernel[dy+pad][dx+pad];
                    }
                }
            }
            // 数据控制在0-255之间
            v = fmax(v, 0);
            v = fmin(v, 255);
            out.at<uchar>(y, x) = (uchar)v;
        }
    }
    return out;
}
int main(int argc, const char* argv[]){
    // read image
    cv::Mat gray = cv::imread("../imori.jpg", cv::IMREAD_GRAYSCALE);

    // diff filter (vertical)
    cv::Mat out_v = diff_filter(gray, 3, false);

    // diff filter (horizontal)
    cv::Mat out_h = diff_filter(gray, 3, true);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_14_vertical.jpg", out_v);
    cv::imwrite("answer_14_horizontal.jpg", out_h);
    return 0;
}