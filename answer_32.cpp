#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <complex>

using namespace cv;
using namespace std;

const int height = 128, width = 128;

struct fourier_str {
    complex<double> coef[height][width];//复数
};

fourier_str dft(Mat img, fourier_str fourier_s) {
    double I;
    double theta;
    complex<double> val;
    for (int l = 0; l < height; l++) {
        for (int k = 0; k < width; k++) {
            val.real(0);
            val.imag(0);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    I = (double) img.at<uchar>(y, x);
                    theta = -2 * M_PI *
                            ((double) k * (double) x / (double) width +
                             (double) l * (double) y / (double) height);
                    val += complex<double>(cos(theta), sin(theta)) * I;
                }
            }
            val /= sqrt(height * width);
            fourier_s.coef[l][k] = val;
        }
    }
    return fourier_s;
}

Mat idft(Mat out,fourier_str fourier_s){
    double theta;
    double g;

    complex<double >G;
    complex<double >val;

    for ( int y = 0; y < height; y ++){
        for ( int x = 0; x < width; x ++){
            val.real(0);
            val.imag(0);
            for ( int l = 0; l < height; l ++){
                for ( int k = 0; k < width; k ++){
                    G = fourier_s.coef[l][k];
                    theta = 2 * M_PI * ((double)k * (double)x / (double)width +
                            (double)l * (double)y / (double)height);
                    val += complex<double>(cos(theta), sin(theta)) * G;
                }
            }
            g = abs(val) / sqrt(height * width);
            out.at<uchar>(y, x) = (uchar)g;
        }
    }
    return out;
}

int main(int argc, const char* argv[]){

    // read original image
    cv::Mat gray = cv::imread("../imori.jpg", cv::IMREAD_GRAYSCALE);

    // Fourier coefficient
    fourier_str fourier_s;

    // output image
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

    // DFT
    fourier_s = dft(gray, fourier_s);

    // IDFT
    out = idft(out, fourier_s);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_32.jpg", out);
    return 0;
}
