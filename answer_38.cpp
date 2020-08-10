#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <complex>

using namespace cv;
using namespace std;

const int height = 128, width = 128, channel = 3;
int T = 8, K = 4;


struct dct_str {
    double coef[height][width][channel];
};

dct_str dct(Mat img, dct_str dct_s) {
    double I, F, Cu, Cv;

    for (int ys = 0; ys < height; ys += T) {
        for (int xs = 0; xs < width; xs += T) {
            for (int c = 0; c < channel; c++) {
                for (int v = 0; v < T; v++) {
                    for (int u = 0; u < T; u++) {
                        F = 0;
                        if (u == 0) {
                            Cu = 1. / sqrt(2);
                        } else {
                            Cu = 1;
                        }

                        if (v == 0) {
                            Cv = 1. / sqrt(2);
                        } else {
                            Cv = 1;
                        }

                        for (int y = 0; y < T; y++) {
                            for (int x = 0; x < T; x++) {
                                I = (double) img.at<Vec3b>(ys + y, xs + x)[c];
                                F += 2. / T * Cu * Cv * I * cos((2. * x + 1) * u * M_PI / 2. / T) *
                                     cos((2. * y + 1) * v * M_PI / 2. / T);
                            }
                        }
                        dct_s.coef[ys + v][xs + u][c] = F;
                    }
                }
            }
        }
    }
    return dct_s;
}

Mat idct(cv::Mat out, dct_str dct_s) {
    double f, Cu, Cv;
    for (int ys = 0; ys < height; ys += T) {
        for (int xs = 0; xs < width; xs += T) {
            for (int c = 0; c < channel; c++) {
                for (int y = 0; y < T; y++) {
                    for (int x = 0; x < T; x++) {
                        f = 0;

                        for (int v = 0; v < K; v++) {
                            for (int u = 0; u < K; u++) {
                                if (u == 0) {
                                    Cu = 1. / sqrt(2);
                                } else {
                                    Cu = 1;
                                }

                                if (v == 0) {
                                    Cv = 1. / sqrt(2);
                                } else {
                                    Cv = 1;
                                }
                                f += 2. / T * Cu * Cv * dct_s.coef[ys + v][xs + u][c] *
                                     cos((2. * x + 1) * u * M_PI / 2. / T) * cos((2. * y + 1) * v * M_PI / 2. / T);
                            }
                        }

                        f = fmin(fmax(f, 0), 255);
                        out.at<cv::Vec3b>(ys + y, xs + x)[c] = (uchar) f;
                    }
                }
            }
        }
    }

    return out;
}

dct_str quantization(dct_str dct_s) {
    double Q[T][T] = {{16, 11, 10, 16, 24,  40,  51,  61},
                      {12, 12, 14, 19, 26,  58,  60,  55},
                      {12, 12, 14, 19, 26,  58,  60,  55},
                      {14, 17, 22, 29, 51,  87,  80,  62},
                      {18, 22, 37, 56, 68,  109, 103, 77},
                      {24, 35, 55, 64, 81,  104, 113, 92},
                      {49, 64, 78, 87, 103, 121, 120, 101},
                      {72, 92, 95, 98, 112, 100, 103, 99}
    };
    for (int ys = 0; ys < height; ys += T) {
        for (int xs = 0; xs < width; xs += T) {
            for (int c = 0; c < channel; c++) {
                for (int y = 0; y < T; y++) {
                    for (int x = 0; x < T; x++) {
                        dct_s.coef[ys + y][xs + x][c] = round(dct_s.coef[ys + y][xs + x][c] / Q[y][x]) * Q[y][x];
                    }
                }
            }
        }
    }
    return dct_s;
}

double MSE(Mat img1, Mat img2) {
    double mse = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channel; c++) {
                mse += pow(((double) img1.at<cv::Vec3b>(y, x)[c] - (double) img2.at<cv::Vec3b>(y, x)[c]), 2);
            }
        }
    }

    mse /= (height * width);
    return mse;
}

double PSNR(double mse, double v_max) {
    return 10 * log10(v_max * v_max / mse);
}


double BITRATE() {
    return T * K * K / T * T;
}

int main(int argc, const char *argv[]) {

    double mse;
    double psnr;
    double bitrate;

    // read original image
    Mat img = imread("../imori.jpg", cv::IMREAD_COLOR);

    // DCT coefficient
    dct_str dct_s;

    // output image
    Mat out = Mat::zeros(height, width, CV_8UC3);

    // DCT
    dct_s = dct(img, dct_s);

    // Quantization
    dct_s = quantization(dct_s);

    // IDCT
    out = idct(out, dct_s);

    // MSE, PSNR
    mse = MSE(img, out);
    psnr = PSNR(mse, 255);
    bitrate = BITRATE();

    std::cout << "MSE: " << mse << std::endl;
    std::cout << "PSNR: " << psnr << std::endl;
    std::cout << "bitrate: " << bitrate << std::endl;
    imwrite("answer_38.jpg", out);
    return 0;
}
