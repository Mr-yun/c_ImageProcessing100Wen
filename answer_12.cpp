//
// Created by YunShao on 2020/7/14.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat motion_filter(Mat img, int kernek_size) {
    int height = img.rows, width = img.cols, channel = img.channels();
    Mat out = Mat::zeros(height, width, CV_8UC3);

    int pad = floor(kernek_size / 2);
    double kernel[kernek_size][kernek_size];

    for (int y = 0; y < kernek_size; y++) {
        for (int x = 0; x < kernek_size; x++) {
            if (y == x) {
                kernel[y][x] = 1 / kernek_size;
            } else {
                kernel[y][x] = 0;
            }
        }
    }

    double v=0;
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            for(int c=0;c<channel;c++){
                v = 0;
                for(int dy=-pad;dy<pad+1;dy++){
                    for(int dx=-pad;dx<pad+1;dx++){
                        if (((y + dy) >= 0) && (( x + dx) >= 0) && ((y + dy) < height) && ((x + dx) < width)){
                            v += (double)img.at<cv::Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
                        }
                    }
                }
            }
        }

    }
    return out;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);

    // motion filter
    cv::Mat out = motion_filter(img, 3);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_12.jpg", out);

    return 0;
}