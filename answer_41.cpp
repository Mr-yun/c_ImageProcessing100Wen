#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

Mat gaussian_filter(Mat img, double sigma, int kernel_size) {
    int height = img.rows,
            width = img.cols,
            channel = img.channels();
    Mat out;
    if (channel == 1) {
        out = Mat::zeros(height, width, CV_8SC1);
    } else {
        out = Mat::zeros(height, width, CV_8SC3);
    }

    int pad = floor(kernel_size / 2);
    int _x = 0, _y = 0;
    double kernel_sum = 0;
    float kernel[kernel_size][kernel_size];

    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            _y = y - pad;
            _x = x - pad;
            kernel[y][x] = 1 / (2 * M_PI * sigma * sigma) * exp(_x * _x + _y * _y) / (2 * sigma * sigma);
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
                        if (((x + dx) >= 0 && (y + dy) >= 0) && ((x + dx) < width) && ((y + dy) < height)) {
                            v += (double) img.at<Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
                        }
                    }
                }
                if (channel == 3) {
                    out.at<Vec3b>(y, x)[c] = (u_char) fmin(fmax(v, 0), 255);
                } else {
                    out.at<uchar>(y, x) = (uchar) fmin(fmax(v, 0), 255);
                }
            }

        }
    }
    return out;
}

Mat sobel_filter(Mat img, int kernel_size, bool horizontal) {
    int height = img.rows,
            width = img.cols,
            channel = img.channels();

    Mat out = Mat::zeros(height, width, CV_8UC1);
    double kernel[kernel_size][kernel_size] = {{1,  2,  1},
                                               {0,  0,  0},
                                               {-1, -2, -1}};

    if (horizontal) {
        kernel[0][1] = 0;
        kernel[0][2] = -1;
        kernel[1][0] = 2;
        kernel[1][2] = -2;
        kernel[2][0] = 1;
        kernel[2][1] = 0;
    }

    int pad = floor(kernel_size / 2);
    double v = 0;
    // filtering
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            v = 0;
            for (int dy = -pad; dy < pad + 1; dy++) {
                for (int dx = -pad; dx < pad + 1; dx++) {
                    if (((y + dy) >= 0) && ((x + dx) >= 0) && ((y + dy) < height) && ((x + dx) < width)) {
                        v += (double) img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
                    }
                }
            }
            out.at<uchar>(y, x) = (uchar) fmin(fmax(v, 0), 255);
        }
    }
    return out;
}

Mat get_edge(Mat fx, Mat fy) {
    int height = fx.rows;
    int width = fx.cols;
    Mat out = Mat::zeros(height, width, CV_8UC1);

    double _fx, _fy;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            _fx = (double) fx.at<uchar>(y, x);
            _fy = (double) fy.at<uchar>(y, x);

            out.at<uchar>(y, x) = (uchar) fmin(fmax(sqrt(_fx * _fx + _fy * _fy), 0), 255);
        }
    }
    return out;
}

Mat get_angle(cv::Mat fx, cv::Mat fy) {
    int height = fx.rows;
    int width = fx.cols;

    // prepare output
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

    double _fx, _fy;
    double angle;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            _fx = fmax((double) fx.at<uchar>(y, x), 0.000001);
            _fy = (double) fy.at<uchar>(y, x);

            angle = atan2(_fy, _fx);
            angle = angle / M_PI * 180;

            if (angle < -22.5) {
                angle = 180 + angle;
            } else if (angle >= 157.5) {
                angle = angle - 180;
            }

            if (angle <= 22.5){
                out.at<uchar>(y, x) = 0;
            } else if (angle <= 67.5){
                out.at<uchar>(y, x) = 45;
            } else if (angle <= 112.5){
                out.at<uchar>(y, x) = 90;
            } else {
                out.at<uchar>(y, x) = 135;
            }
        }
    }
    return out;
}

// Canny step 1
int Canny_step1(cv::Mat gray){

    // gaussian filter
    cv::Mat gaussian = gaussian_filter(gray, 1.4, 5);

    // sobel filter (vertical)
    cv::Mat fy = sobel_filter(gaussian, 3, false);

    // sobel filter (horizontal)
    cv::Mat fx = sobel_filter(gaussian, 3, true);

    // get edge
    cv::Mat edge = get_edge(fx, fy);

    // get angle
    cv::Mat angle = get_angle(fx, fy);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_edge_41.jpg", edge);
    cv::imwrite("answer_angle_41.jpg", angle);

    return 0;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_GRAYSCALE);

    // Canny step 1
    Canny_step1(img);

    return 0;
}