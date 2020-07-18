#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
//直方图归一化 将直方图所有分量限制在一定范围
using namespace cv;
using namespace std;

Mat histogram_normalization(cv::Mat img, int a, int b) {
    int height = img.cols, width = img.rows, channel = img.channels();
    int c=INT8_MAX, d=0;
    int val;

    Mat out = Mat::zeros(height, width, CV_8UC3);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int _c = 0; _c < channel; _c++) {
                val = (float) img.at<Vec3b>(y, x)[_c];
                c = fmin(c, val);
                d = fmax(d, val);
            }
        }
    }
    cout << c << "\t" << d << endl;
    // histogram transformation
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int _c = 0; _c < channel; _c++) {
                val = (float) img.at<Vec3b>(y, x)[_c];
                if (val < a) out.at<Vec3b>(y, x)[_c] = (uchar) val;
                else if (val > a && val < b) out.at<Vec3b>(y, x)[_c] = (uchar) ((b - a) / (d - c) * (val - c) + a);
                else out.at<cv::Vec3b>(y, x)[_c] = (uchar) b;
            }
        }
    }

    return out;
}

int main(int argc, const char *argv[]) {
    // read image
    cv::Mat img = cv::imread("../imori_dark.jpg", cv::IMREAD_COLOR);

    // histogram normalization
    cv::Mat out = histogram_normalization(img, 0, 255);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_21.jpg", out);
    return 0;
}