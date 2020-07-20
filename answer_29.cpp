#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
// 作者方法没看懂,这里结合样例25,进行修改
Mat affine(cv::Mat img, double rx, double ry, double tx, double ty) {
    // get height and width
    int width = img.cols;
    int height = img.rows;
    int channel = img.channels();

    int resized_width = (int) (width * rx);
    int resized_height = (int) (height * ry);

    int x_before, y_before;

    Mat out = Mat::zeros(resized_height, resized_width, CV_8UC3);

    for (int y = 0; y < resized_height; y++) {
        y_before = (int) round(y / ry);
        y_before = fmin(y_before, height - 1);
        if (y + ty <= 0 || y + ty > resized_height) continue;

        for (int x = 0; x < resized_width; x++) {
            x_before = (int) round(x / rx);
            x_before = fmin(x_before, width - 1);
            if (x + tx <= 0 || x + tx > resized_width) continue;

            for (int c = 0; c < channel; c++) {
                out.at<Vec3b>(y + ty, x + tx)[c] = img.at<Vec3b>(y_before, x_before)[c];
            }
        }
    }
    return out;
}

int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori.jpg", cv::IMREAD_COLOR);

    // affine
    cv::Mat out = affine(img, 1.3, 0.8, 30, -30);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_29.jpg", out);

    return 0;
}
