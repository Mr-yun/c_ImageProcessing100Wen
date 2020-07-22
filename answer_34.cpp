#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <complex>

const int height = 128, width = 128;
using namespace cv;
struct fourier_str {
    std::complex<double> coef[height][width];
};

fourier_str dft(cv::Mat img, fourier_str fourier_s){

    double I;
    double theta;
    std::complex<double> val;

    for ( int l = 0; l < height; l ++){
        for ( int k = 0; k < width; k ++){
            val.real(0);
            val.imag(0);
            for ( int y = 0; y < height; y ++){
                for ( int x = 0; x < width; x ++){
                    I = (double)img.at<uchar>(y, x);
                    theta = -2 * M_PI * ((double)k * (double)x / (double)width + (double)l * (double)y / (double)height);
                    val += std::complex<double>(cos(theta), sin(theta)) * I;
                }
            }
            val /= sqrt(height * width);
            fourier_s.coef[l][k] = val;
        }
    }

    return fourier_s;
}

Mat idft(cv::Mat out, fourier_str fourier_s){

    double theta;
    double g;
    std::complex<double> G;
    std::complex<double> val;

    for ( int y = 0; y < height; y ++){
        for ( int x = 0; x < width; x ++){
            val.real(0);
            val.imag(0);
            for ( int l = 0; l < height; l ++){
                for ( int k = 0; k < width; k ++){
                    G = fourier_s.coef[l][k];
                    theta = 2 * M_PI * ((double)k * (double)x / (double)width + (double)l * (double)y / (double)height);
                    val += std::complex<double>(cos(theta), sin(theta)) * G;
                }
            }
            g = std::abs(val) / sqrt(height * width);
            g = fmin(fmax(g, 0), 255);
            out.at<uchar>(y, x) = (uchar)g;
        }
    }

    return out;
}

fourier_str hpf(fourier_str fourier_s, double pass_r){

    // filtering
    int r = height / 2;
    int filter_d = (int)((double)r * pass_r);
    for ( int j = 0; j < height / 2; j++){
        for ( int i = 0; i < width / 2; i++){
            if (sqrt(i * i + j * j) <= filter_d){
                fourier_s.coef[j][i] = 0;
                fourier_s.coef[j][width - i] = 0;
                fourier_s.coef[height - i][i] = 0;
                fourier_s.coef[height - i][width - i] = 0;
            }
        }
    }
    return fourier_s;
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

    // HPF
    fourier_s = hpf(fourier_s, 0.1);

    // IDFT
    out = idft(out, fourier_s);

    //cv::imwrite("out.jpg", out);
    cv::imwrite("answer_34.jpg", out);

    return 0;
}

