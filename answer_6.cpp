//
// Created by YunShao on 2020/7/12.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat decrease_color(Mat img) {
    int height = img.cols,
            width = img.rows,
            channel = img.channels();

    Mat out = Mat::zeros(height, width, CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channel; c++) {
                out.at<Vec3b>(y, x)[c] = (uchar) (floor((double) img.at<Vec3b>(y, x)[c] / 64) * 64 + 32);
            }
        }
    }
    return out;
}
int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);

    // decrease color
    cv::Mat out = decrease_color(img);

    cv::imwrite("answer6.jpg", out);

    return 0;
}
