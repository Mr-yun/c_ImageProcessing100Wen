#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
//伽马校正用来对照相机等电子设备传感器的非线性光电转换特性进行校正。如果图像原样显示在显示器等上，画面就会显得很暗。
// 伽马校正通过预先增大 RGB 的值来排除显示器的影响，达到对图像修正的目的
using namespace cv;

Mat gamma_correction(cv::Mat img, double gamma_c, double gamma_g){
    // get height and width
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    // output image
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);

    double val;

    // gamma correction
    for (int y = 0; y< height; y++){
        for (int x = 0; x < width; x++){
            for (int c = 0; c < channel; c++){
                val = (double)img.at<Vec3b>(y,x)[c]/255;
                out.at<cv::Vec3b>(y, x)[c] = (uchar)(pow(val/gamma_c,1/gamma_g)*255);
            }
        }
    }

    return out;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori_gamma.jpg", cv::IMREAD_COLOR);

    // gamma correction
    cv::Mat out = gamma_correction(img, 1, 2.2);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_24.jpg", out);

    return 0;
}