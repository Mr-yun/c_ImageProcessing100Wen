//
// Created by YunShao on 2020/7/12.
// 通过自定义二值化
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "iostream"

using namespace cv;

Mat Binarize(Mat gray, int th) {
    int width = gray.cols, height = gray.rows;
    Mat out = Mat::zeros(width, height, CV_8UC1);

    for(int y=0;y<height;y++){
        for(int x=0;x<width ;x++){
            if(gray.at<uchar >(y,x)>th){
                out.at<uchar >(y,x)=255;
            } else{
                out.at<uchar >(y,x)=0;
            }
        }
    }
    return out;
};

int main(int argc, const char* argv[]){
    // read image
    Mat img = imread("../imori.jpg", IMREAD_GRAYSCALE);

    // Gray -> Binary
    Mat out = Binarize(img, 128);

    //cv::imwrite("out.jpg", out);
    imwrite("answer3.jpg", out);

    return 0;
}