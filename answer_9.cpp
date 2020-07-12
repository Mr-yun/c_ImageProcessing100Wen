#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat gaussian_filter(Mat img, double sigma, int kernel_size) {
    int height = img.rows,
            width = img.cols,
            channel = img.channels();
    Mat out = Mat::zeros(height, width, CV_8UC3);

    int pad = floor(kernel_size / 2);
    int _x = 0, _y = 0;
    double kernel_sum = 0;

    float kernel[kernel_size][kernel_size];

    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            _y = y - pad;
            _x = x - pad;
            kernel[y][x] = 1 / (2 * M_PI * sigma * sigma) * exp(-(_x * _x + _y * _y) / (2 * sigma * sigma));
            kernel_sum += kernel[y][x];
        }
    }

    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            kernel[y][x] /= kernel_sum;
        }
    }

    double v = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channel; c++) {
                v = 0;
                for (int dy = -pad; dy < pad + 1; dy++) {
                    for (int dx = -pad; dx < pad + 1; dx++) {
                        if (((x + dx) >= 0) && ((y + dy) >= 0)) {
                            v += (double) img.at<cv::Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
                        }
                    }
                }
                out.at<Vec3b>(y, x)[c] = v;
            }

        }
    }
    return out;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori_noise.jpg", cv::IMREAD_COLOR);

    // gaussian filter
    cv::Mat out = gaussian_filter(img, 1.3, 3);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer9.jpg", out);

    return 0;
}