//
// Created by YunShao on 2020/7/13.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;
Mat median_filter(Mat img, int kernel_size) {
    int height = img.rows,
            width = img.cols,
            channle = img.channels();

    Mat out = Mat::zeros(height, width, CV_8UC3);
    int pad = floor(kernel_size / 2);
    double v = 0;
    int vs[kernel_size * kernel_size];
    int count = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channle; c++) {
                v = 0;
                count = 0;
                for (int i = 0; i < kernel_size * kernel_size; i++) {
                    vs[i] = 999;
                }

                for(int dy=-pad;dy<pad+1;dy++){
                    for(int dx=-pad;dx<pad+1;dx++){
                        if (((y+dy)>=0)&&((x+dx)>=0)){
                            vs[count++] = (int)img.at<Vec3b>(y+dy,x+dx)[c];
                        }
                    }
                }

                sort(vs,vs+(kernel_size*kernel_size));
                out.at<Vec3b>(y,x)[c] = (uchar)vs[int(floor(count/2))+1];
            }
        }
    }
    return out;
}
int main(int argc, const char* argv[]){
    // read image
    cv::Mat img = cv::imread("../imori_noise.jpg", cv::IMREAD_COLOR);

    cv::Mat out = median_filter(img, 3);

    cv::imwrite("answer_10.jpg", out);

    return 0;
}