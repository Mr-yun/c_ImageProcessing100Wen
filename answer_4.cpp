#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;

Mat Binarize_Otsu(Mat gray) {
    int width = gray.cols, height = gray.rows;
    double w0 = 0, w1 = 0;
    double m0 = 0, m1 = 0;
    double max_sb = 0, sb = 0;
    int th = 0;
    int val;

    for (int t = 0; t < 255; t++) {
        w0 = 0;
        w1 = 0;
        m0 = 0;
        m1 = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                val = int(gray.at<uchar>(y, x));
                if (val < t) {
                    w0++;
                    m0 += val;
                } else {
                    w1++;
                    m1 += val;
                }
            }
        }

        m0 /= w0; //两个类的像素值的平均值
        m1 /= w1;

        w0 /= (width * height); //被阈值t分开的两个类中的像素数占总像素数的比率
        w1 /= (width * height); //类间方差

        sb = w0 * w1 * pow((m0 - m1), 2);
        if (sb > max_sb) {
            max_sb = sb;
            th = t;
        }
    }

    std::cout << "threshold:" << th << std::endl;
    // prepare output
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
    // each y, x
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Binarize
            if (gray.at<uchar>(y, x) > th) {
                out.at<uchar>(y, x) = 255;
            } else {
                out.at<uchar>(y, x) = 0;
            }

        }
    }

    return out;
}

int main(int argc, const char *argv[]) {
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_GRAYSCALE);


    // Gray -> Binary
    cv::Mat out = Binarize_Otsu(img);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer4.jpg", out);

    return 0;
}