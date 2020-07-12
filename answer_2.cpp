//
// Created by YunShao on 2020/7/12.
// 灰度化
// 灰度图像即图像3通道不同权重的值
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

//todo gpu如何加速

Mat BGR2GRAY(Mat img) {
    int width = img.cols, height = img.rows;
    Mat out = Mat::zeros(height,width,CV_8UC1);
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            out.at<uchar>(y,x) = 0.216 * (float) img.at<Vec3b>(y,x)[2] +
                    0.7152   * (float)img.at<Vec3b>(y,x)[1] +
                    0.0722 * (float)img.at<cv::Vec3b>(y, x)[0];
        }
    }
    return out;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);

    // BGR -> Gray
    cv::Mat out = BGR2GRAY(img);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("sample.jpg", out);
    return 0;
}