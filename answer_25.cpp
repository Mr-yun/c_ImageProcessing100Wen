#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat nearest_neighbor(cv::Mat img, double rx, double ry) {
    int width = img.cols, height = img.rows, channle = img.channels();

    int resized_width = (int) (width * rx);
    int resized_height = (int) (height * ry);

    int x_before, y_before;

    Mat out = Mat::zeros(resized_height, resized_width, CV_8UC3);

    for (int y = 0; y < resized_height; y++) {
        y_before = (int) round(y / ry);
        y_before = fmin(y_before, height - 1);

        for (int x = 0; x < resized_width; x++) {
            x_before = (int) round(x / rx);
            x_before = fmin(x_before, width - 1);

            for (int c = 0; c < channle; c++) {
                out.at<Vec3b>(y, x)[c] = img.at<Vec3b>(y_before, x_before)[c];
            }
        }
    }
    return out;
}

int main(int argc, const char *argv[]) {
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);
    cv::Mat out = nearest_neighbor(img, 1.5, 1.5);
    cv::imwrite("answer_25.jpg", out);

    return 0;
}