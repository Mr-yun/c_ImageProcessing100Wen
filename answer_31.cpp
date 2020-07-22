#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

//
void new_center(int width, int height,
                double x_1, double x_2,
                double y_1, double y_2,
                int *tx, int *ty) {
    // 计算中心点,计算偏移量
    // 使得最后中心点依然落在原来中心点,使得图像边缘最小裁剪
    double cx = width / 2.;
    double cy = height / 2.;
    double new_cx = x_1 * cx - x_2 * cy;
    double new_cy = y_1 * cx + y_2 * cy;

    *tx = (int) (new_cx - cx);
    *ty = (int) (new_cy - cy);
}

Mat affine(cv::Mat img, double dx, double dy) {
    // get height and width
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    // other parameters
    int x_new, y_new;

    Mat out = cv::Mat::zeros(height + dy, width + dx, CV_8UC3);
    double a_x = dy / width, a_y = dx / height;
    cout << a_x << "\t" << a_y << endl;
    // Affine transformation
    for (int y = 0; y < out.rows; y++) {
        for (int x = 0; x < out.cols; x++) {
            // get original position y
            y_new = y - a_x * x;
            if ((y_new < 0) || (y_new >= height)) {
                continue;
            }

            // get original position x
            x_new = x - a_y * y;
            if ((x_new < 0) || (x_new >= width)) {
                continue;
            }

            // assign pixel to new position
            for (int c = 0; c < channel; c++) {
                out.at<cv::Vec3b>(y, x)[c] = img.at<cv::Vec3b>(y_new, x_new)[c];
            }
        }
    }

    return out;
}

int main(int argc, const char *argv[]) {
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);
    // affine
    cv::Mat out = affine(img, 30, 30);
    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_31.jpg", out);

    return 0;
}