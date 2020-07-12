#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat average_pooling(Mat img, int r = 8) {
    int height = img.rows,
            width = img.cols,
            channel = img.channels();

    Mat out = Mat::zeros(height, width, CV_8UC3);

    double v = 0;

    for (int y = 0; y < height; y += r) {
        for (int x = 0; x < width; x += r) {
            for (int c = 0; c < channel; c++) {
                v = 0;
                for (int dy = 0; dy < r; dy++) {
                    for (int dx = 0; dx < r; dx++) {
                        v = fmax(img.at<Vec3b>(y+dy,x+dx)[c],v);
                    }
                }
                v /= (r * r);
                for (int dy = 0; dy < r; dy++) {
                    for (int dx = 0; dx < r; dx++) {
                        out.at<Vec3b>(y + dy, x + dx)[c] = (uchar) v;
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

    // average pooling
    cv::Mat out = average_pooling(img);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer8.jpg", out);
    return 0;
}