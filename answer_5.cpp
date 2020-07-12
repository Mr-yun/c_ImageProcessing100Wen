#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat BGR2HSV(Mat img) {
    int width = img.cols, height = img.rows;
    float r, g, b;
    float h, s, v;
    float _max, _min;

    Mat hsv = Mat::zeros(height, width, CV_32FC3);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            r = (float) img.at<Vec3b>(y, x)[2] / 255;
            g = (float) img.at<Vec3b>(y, x)[1] / 255;
            b = (float) img.at<Vec3b>(y, x)[0] / 255;

            _max = fmax(r, fmax(g, b));
            _min = fmin(r, fmin(g, b));

            if (_max == _min) {
                h = 0;
            } else if (_min == b) {
                h = 60 * (g - r) / (_max - _min) + 60;
            } else if (_min == r) {
                h = 60 * (b - g) / (_max - _min) + 180;
            } else {
                h = 60 * (r - b) / (_max - _min) + 300;
            }

            s = _max - _min;
            v = _max;

            hsv.at<Vec3f>(y, x)[0] = h;
            hsv.at<Vec3f>(y, x)[1] = s;
            hsv.at<Vec3f>(y, x)[2] = v;
        }
    }
    return hsv;
}

Mat HSV2BGR(Mat hsv) {
    int width = hsv.cols, height = hsv.rows;
    float h, s, v;
    double c, _h, _x;
    double r, g, b;
    Mat out = Mat::zeros(height, width, CV_8UC3);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h = hsv.at<Vec3f>(y, x)[0];
            s = hsv.at<Vec3f>(y, x)[1];
            v = hsv.at<Vec3f>(y, x)[2];

            c = s;
            _h = h / 60;
            _x = c * (1 - abs(fmod(_h, 2) - 1));

            r = g = b = v - c;

            if (_h < 1) {
                r += c;
                g += _x;
            } else if (_h < 2) {
                r += _x;
                g += c;
            } else if (_h < 3) {
                g += c;
                b += _x;
            } else if (_h < 4) {
                g += _x;
                b += c;
            } else if (_h < 5) {
                r += _x;
                b += c;
            } else if (_h < 6) {
                r += c;
                b += _x;
            }

            out.at<Vec3b>(y, x)[0] = (uchar) (b * 255);
            out.at<Vec3b>(y, x)[1] = (uchar) (g * 255);
            out.at<Vec3b>(y, x)[2] = (uchar) (r * 255);
        }
    }
    return out;
}

Mat inverse_hue(Mat hsv) {
    int height = hsv.rows, width = hsv.cols;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            hsv.at<Vec3f>(y, x)[0] = fmod(hsv.at<Vec3f>(y, x)[0] + 180, 360);
        }
    }
    return hsv;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);

    // BGR -> HSV
    cv::Mat hsv = BGR2HSV(img);

    // Inverse Hue
    hsv = inverse_hue(hsv);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer5.jpg", hsv);
    return 0;
}