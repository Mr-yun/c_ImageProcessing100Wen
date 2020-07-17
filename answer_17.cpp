#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
//Laplacian滤波器是对图像亮度进行二次微分从而检测边缘的滤波器

using namespace cv;

Mat laplacian_filter(cv::Mat img, int kernel_size) {
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();

    // prepare output
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

    // prepare kernel
    double kernel[kernel_size][kernel_size] = {{0, 1,  0},
                                               {1, -4, 1},
                                               {0, 1,  0}};

    int pad = floor(kernel_size / 2);

    double v = 0;

    // filtering
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            v = 0;
            for (int dy = -pad; dy < pad + 1; dy++) {
                for (int dx = -pad; dx < pad + 1; dx++) {
                    if (((y + dy) >= 0) && ((x + dx) >= 0) && ((y + dy) < height) && ((x + dx) < width)) {
                        v += img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
                    }
                }
            }
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

    // laplacian filter
    cv::Mat out = laplacian_filter(gray, 3);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_17.jpg", out);

    return 0;
}
