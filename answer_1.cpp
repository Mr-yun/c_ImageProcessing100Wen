//
// Created by YunShao on 2020/7/12.
// 读取图像，然后将RGB通道替换成BGR通道。
// 使用mat.at<Vec3b>(h,w)[channel] 进行转换
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

Mat channel_swap(Mat img) {
    int width = img.cols,
            height = img.rows;
    Mat out = Mat::zeros(height, width, CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            out.at<Vec3b>(y, x)[0] = img.at<Vec3b>(y, x)[2];
            out.at<Vec3b>(y, x)[2] = img.at<Vec3b>(y, x)[0];
            out.at<Vec3b>(y, x)[1] = img.at<Vec3b>(y, x)[1];
        }
    }
    return out;
}

int main(int argc, const char *argv[]) {
    Mat img = imread("../imori.jpg");
    Mat out = channel_swap(img);
    imwrite("answer.jpg", out);
    return 0;
}