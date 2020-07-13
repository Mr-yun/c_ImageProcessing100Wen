//
// Created by YunShao on 2020/7/13.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat mean_filter(Mat img, int kernel_size) {
    int height = img.rows,
            width = img.cols,
            channel = img.channels();

    Mat out = Mat::zeros(height, width, CV_8UC3);
    int pad = floor(kernel_size / 2);

    double v = 0;
    int kernel_size_s = kernel_size * kernel_size;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channel; c++) {
                v = 0;

                for (int dy = -pad; dy < pad + 1; dy++) {
                    for (int dx = -pad; dx < pad + 1; dx++) {
                        if ((y + dy) >= 0 && (x + dx) >= 0) {
                            v += (int) img.at<Vec3b>(y+dy, x+dx)[c];
                        }
                    }
                }

                v /= kernel_size_s;
                out.at<Vec3b>(y,x)[c] = (uchar)v;
            }
        }
    }
    return out;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);

    // mean filter
    cv::Mat out = mean_filter(img, 3);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_11.jpg", out);

    return 0;
}