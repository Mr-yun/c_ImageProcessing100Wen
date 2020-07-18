#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

Mat histogram_transform(cv::Mat img, int m0, int s0) {
    // get height and width
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    // histogram transformation hyper-parameters
    double m, s;
    double sum = 0., squared_sum = 0.;
    double val;

    // output image
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);

    // get sum
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channel; c++) {
                val = (float) img.at<Vec3b>(y, x)[c];
                sum += val;
                squared_sum += (val * val);
            }
        }
    }

    // get standard deviation
    m = sum / (height * width * channel); //平均值
    s = sqrt(squared_sum / (height * width * channel) - m * m); // 标准差


    // histogram transformation
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                val = img.at<Vec3b>(y, x)[c];
                out.at<Vec3b>(y, x)[c] = (uchar) (s0 / s * (val - m) + m0);
            }
        }
    }

    return out;
}

int main(int argc, const char *argv[]) {
    // read image
    cv::Mat img = cv::imread("../imori_dark.jpg", cv::IMREAD_COLOR);

    // histogram transform
    cv::Mat out = histogram_transform(img, 128, 52);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_22.jpg", out);

    return 0;
}