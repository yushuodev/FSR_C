#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace cv;

#define SHARPNESS 0.1f
#define FSR_RCAS_DENOISE 1 
#define FSR_PQ 0 
#define FSR_RCAS_LIMIT (0.25f - (1.0f / 16.0f)) 

float APrxRcp(float a) {
    return 1 / (a + 0.0001f);
}

float AMax3F1(float x, float y, float z) {
    return std::max(x, std::max(y, z));
}

float AMin3F1(float x, float y, float z) {
    return std::min(x, std::min(y, z));
}

void FsrEasuTap(
    Vec3f& aC,  // Accumulated color, with negative lobe.
    float& aW,  // Accumulated weight.
    Point2f off, // Pixel offset from resolve position to tap.
    Point2f dir, // Gradient direction.
    Point2f len, // Length.
    float lob,   // Negative lobe strength.
    float clp,   // Clipping point.
    Vec3f c) {   // Tap color.

    // Rotate offset by direction.
    Point2f v = Vec2f(0.0f, 0.0f);
    v.x = (off.x * (dir.x)) + (off.y * dir.y);
    v.y = (off.x * (-dir.y)) + (off.y * dir.x);

    // Anisotropy.
    v.x = v.x * len.x;
    v.y = v.y * len.y;

    float d2 = v.x * v.x + v.y * v.y;
    d2 = std::min(d2, clp);
    float wB = 2.0f / 5.0f * d2 - 1.0f;
    float wA = lob * d2 - 1.0f;
    wB *= wB;
    wA *= wA;
    wB = 25.0f / 16.0f * wB - (25.0f / 16.0f - 1.0f);
    float w = wB * wA;
    aC[0] = aC[0] + c[0] * w;
    aC[1] = aC[1] + c[1] * w;
    aC[2] = aC[2] + c[2] * w;

    aW = aW + w;
}

cv::Vec3f Min4(cv::Vec3f x, cv::Vec3f y, cv::Vec3f z, cv::Vec3f w) {
    return cv::Vec3f(std::min(std::min(x[0], y[0]), std::min(z[0], w[0])),
        std::min(std::min(x[1], y[1]), std::min(z[1], w[1])),
        std::min(std::min(x[2], y[2]), std::min(z[2], w[2])));
}

cv::Vec3f Max4(cv::Vec3f x, cv::Vec3f y, cv::Vec3f z, cv::Vec3f w) {
    return cv::Vec3f(std::max(std::max(x[0], y[0]), std::max(z[0], w[0])),
        std::max(std::max(x[1], y[1]), std::max(z[1], w[1])),
        std::max(std::max(x[2], y[2]), std::max(z[2], w[2])));
}
float clamp(float val, float min, float max)
{
    if (val <= min)
        val = min;
    else if (val >= max)
        val = max;
    else
        val = val;
    return val;
}

void FsrEasuSet(
    Point2f& dir,
    float& len,
    Point2f pp,
    bool biS, bool biT, bool biU, bool biV,
    float lA, float lB, float lC, float lD, float lE) {

    float w = 0.0f;
    if (biS) w = (1.0f - pp.x) * (1.0f - pp.y);
    if (biT) w = pp.x * (1.0f - pp.y);
    if (biU) w = (1.0f - pp.x) * pp.y;
    if (biV) w = pp.x * pp.y;

    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = std::max(std::abs(dc), std::abs(cb));
    float dirX = lD - lB;

    dir.x += dirX * w;

    lenX = clamp(std::abs(dirX) / lenX, 0.0f, 1.0f);
    lenX = lenX * lenX;
    len = len + (lenX * w);

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = std::max(std::abs(ec), std::abs(ca));
    float dirY = lE - lA;

    dir.y += dirY * w;
    lenY = clamp(std::abs(dirY) / lenY, 0.0f, 1.0f);
    lenY = lenY * lenY;
    len = len + (lenY * w);
}

uchar bilinear_interpolation(const Mat& image, float x, float y, int channel) {
    int x1 = (int)x;
    int y1 = (int)y;
    int x2 = min(x1 + 1, image.cols - 1); // Clamp to the maximum valid index
    int y2 = min(y1 + 1, image.rows - 1); // Clamp to the maximum valid index

    float x_diff = x - (float)x1;
    float y_diff = y - (float)y1;

    uchar val1 = (1 - x_diff) * image.at<Vec3b>(y1, x1)[channel] + x_diff * image.at<Vec3b>(y1, x2)[channel];
    uchar val2 = (1 - x_diff) * image.at<Vec3b>(y2, x1)[channel] + x_diff * image.at<Vec3b>(y2, x2)[channel];

    return (uchar)((1 - y_diff) * val1 + y_diff * val2);
}

Vec3b EASU(const Mat& image, float x, float y) {
    Vec2f pp = Vec2f(x, y);
    Vec2f fp = Vec2f(floor(pp[0]), floor(pp[1]));
    pp -= fp;
    int f_x = (int)x;
    int f_y = (int)y;

    float b_b = (float)image.at<Vec3b>(f_y + (-1), f_x + (0))[0];
    float c_b = (float)image.at<Vec3b>(f_y + (-1), f_x + (1))[0];
    float i_b = (float)image.at<Vec3b>(f_y + (1), f_x + (-1))[0];
    float j_b = (float)image.at<Vec3b>(f_y + (1), f_x + (0))[0];
    float f_b = (float)image.at<Vec3b>(f_y + (0), f_x + (0))[0];
    float e_b = (float)image.at<Vec3b>(f_y + (0), f_x + (-1))[0];
    float k_b = (float)image.at<Vec3b>(f_y + (1), f_x + (1))[0];
    float l_b = (float)image.at<Vec3b>(f_y + (1), f_x + (2))[0];
    float h_b = (float)image.at<Vec3b>(f_y + (0), f_x + (2))[0];
    float g_b = (float)image.at<Vec3b>(f_y + (0), f_x + (1))[0];
    float o_b = (float)image.at<Vec3b>(f_y + (2), f_x + (1))[0];
    float n_b = (float)image.at<Vec3b>(f_y + (2), f_x + (0))[0];

    float b_g = (float)image.at<Vec3b>(f_y + (-1), f_x + (0))[1];
    float c_g = (float)image.at<Vec3b>(f_y + (-1), f_x + (1))[1];
    float i_g = (float)image.at<Vec3b>(f_y + (1), f_x + (-1))[1];
    float j_g = (float)image.at<Vec3b>(f_y + (1), f_x + (0))[1];
    float f_g = (float)image.at<Vec3b>(f_y + (0), f_x + (0))[1];
    float e_g = (float)image.at<Vec3b>(f_y + (0), f_x + (-1))[1];
    float k_g = (float)image.at<Vec3b>(f_y + (1), f_x + (1))[1];
    float l_g = (float)image.at<Vec3b>(f_y + (1), f_x + (2))[1];
    float h_g = (float)image.at<Vec3b>(f_y + (0), f_x + (2))[1];
    float g_g = (float)image.at<Vec3b>(f_y + (0), f_x + (1))[1];
    float o_g = (float)image.at<Vec3b>(f_y + (2), f_x + (1))[1];
    float n_g = (float)image.at<Vec3b>(f_y + (2), f_x + (0))[1];

    float b_r = (float)image.at<Vec3b>(f_y + (-1), f_x + (0))[2];
    float c_r = (float)image.at<Vec3b>(f_y + (-1), f_x + (1))[2];
    float i_r = (float)image.at<Vec3b>(f_y + (1), f_x + (-1))[2];
    float j_r = (float)image.at<Vec3b>(f_y + (1), f_x + (0))[2];
    float f_r = (float)image.at<Vec3b>(f_y + (0), f_x + (0))[2];
    float e_r = (float)image.at<Vec3b>(f_y + (0), f_x + (-1))[2];
    float k_r = (float)image.at<Vec3b>(f_y + (1), f_x + (1))[2];
    float l_r = (float)image.at<Vec3b>(f_y + (1), f_x + (2))[2];
    float h_r = (float)image.at<Vec3b>(f_y + (0), f_x + (2))[2];
    float g_r = (float)image.at<Vec3b>(f_y + (0), f_x + (1))[2];
    float o_r = (float)image.at<Vec3b>(f_y + (2), f_x + (1))[2];
    float n_r = (float)image.at<Vec3b>(f_y + (2), f_x + (0))[2];

    float b_l = 0.5 * b_r + b_g + 0.5 * b_b;
    float c_l = 0.5 * c_r + c_g + 0.5 * c_b;
    float i_l = 0.5 * i_r + i_g + 0.5 * i_b;
    float j_l = 0.5 * j_r + j_g + 0.5 * j_b;
    float f_l = 0.5 * f_r + f_g + 0.5 * f_b;
    float e_l = 0.5 * e_r + e_g + 0.5 * e_b;
    float k_l = 0.5 * k_r + k_g + 0.5 * k_b;
    float l_l = 0.5 * l_r + l_g + 0.5 * l_b;
    float h_l = 0.5 * h_r + h_g + 0.5 * h_b;
    float g_l = 0.5 * g_r + g_g + 0.5 * g_b;
    float o_l = 0.5 * o_r + o_g + 0.5 * o_b;
    float n_l = 0.5 * n_r + n_g + 0.5 * n_b;

    Point2f dir = Point2f(0.0, 0.0);
    float len = 0.0;

    FsrEasuSet(dir, len, pp, true, false, false, false, b_l, e_l, f_l, g_l, j_l);
    FsrEasuSet(dir, len, pp, false, true, false, false, c_l, f_l, g_l, h_l, k_l);
    FsrEasuSet(dir, len, pp, false, false, true, false, f_l, i_l, j_l, k_l, n_l);
    FsrEasuSet(dir, len, pp, false, false, false, true, g_l, j_l, k_l, l_l, o_l);
    Vec2f dir2 = Vec2f(dir.x * dir.x, dir.y * dir.y);
    float dirR = dir2[0] + dir2[1];
    bool zro = dirR < 1.0 / 32768.0;
    dirR = 1.0 / sqrt(dirR);
    dirR = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
    dir = Vec2f(dir.x * dirR, dir.y * dirR);
    len = len * 0.5f;
    len = len * len;
    float stretch = (dir.x * dir.x + dir.y * dir.y) / (max(abs(dir.x), abs(dir.y)));
    Vec2f len2 = Vec2f(float(1.0) + (stretch - float(1.0)) * len, float(1.0) + float(-0.5) * len);
    float lob = float(0.5) + float((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = float(1.0) / lob;
    Vec3f min4 = Min4(Vec3f(f_r, f_g, f_b), Vec3f(g_r, g_g, g_b), Vec3f(j_r, j_g, j_b), Vec3f(k_r, k_g, k_b));
    Vec3f max4 = Max4(Vec3f(f_r, f_g, f_b), Vec3f(g_r, g_g, g_b), Vec3f(j_r, j_g, j_b), Vec3f(k_r, k_g, k_b));

    Vec3f aC = Vec3f(0.0, 0.0, 0.0);
    float aW = float(0.0);

    FsrEasuTap(aC, aW, Vec2f(0.0, -1.0) - pp, dir, len2, lob, clp, Vec3f(b_r, b_g, b_b)); // b
    FsrEasuTap(aC, aW, Vec2f(1.0, -1.0) - pp, dir, len2, lob, clp, Vec3f(c_r, c_g, c_b)); // c
    FsrEasuTap(aC, aW, Vec2f(-1.0, 1.0) - pp, dir, len2, lob, clp, Vec3f(i_r, i_g, i_b)); // i
    FsrEasuTap(aC, aW, Vec2f(0.0, 1.0) - pp, dir, len2, lob, clp, Vec3f(j_r, j_g, j_b)); // j
    FsrEasuTap(aC, aW, Vec2f(0.0, 0.0) - pp, dir, len2, lob, clp, Vec3f(f_r, f_g, f_b)); // f
    FsrEasuTap(aC, aW, Vec2f(-1.0, 0.0) - pp, dir, len2, lob, clp, Vec3f(e_r, e_g, e_b)); // e
    FsrEasuTap(aC, aW, Vec2f(1.0, 1.0) - pp, dir, len2, lob, clp, Vec3f(k_r, k_g, k_b)); // k
    FsrEasuTap(aC, aW, Vec2f(2.0, 1.0) - pp, dir, len2, lob, clp, Vec3f(l_r, l_g, l_b)); // l
    FsrEasuTap(aC, aW, Vec2f(2.0, 0.0) - pp, dir, len2, lob, clp, Vec3f(h_r, h_g, h_b)); // h
    FsrEasuTap(aC, aW, Vec2f(1.0, 0.0) - pp, dir, len2, lob, clp, Vec3f(g_r, g_g, g_b)); // g
    FsrEasuTap(aC, aW, Vec2f(1.0, 2.0) - pp, dir, len2, lob, clp, Vec3f(o_r, o_g, o_b)); // o
    FsrEasuTap(aC, aW, Vec2f(0.0, 2.0) - pp, dir, len2, lob, clp, Vec3f(n_r, n_g, n_b)); // n

    Vec3b pix;
    Vec3f clamped_pix;

    clamped_pix[0] = (aC[0] / aW);
    clamped_pix[1] = (aC[1] / aW);
    clamped_pix[2] = (aC[2] / aW);

    for (int i = 0; i < 3; ++i) {
        clamped_pix[i] = std::min(max4[i], std::max(min4[i], clamped_pix[i]));
    }

    pix[2] = static_cast<uchar>(clamped_pix[0]); // Convert the result back to Vec3b
    pix[1] = static_cast<uchar>(clamped_pix[1]);
    pix[0] = static_cast<uchar>(clamped_pix[2]);
    return (pix);
}


Mat resize_bilinear(const Mat& image, int new_width, int new_height) {
    Mat resized_image = Mat::zeros(new_height, new_width, CV_8UC3);
    float x_ratio = (float)image.cols / (float)new_width;
    float y_ratio = (float)image.rows / (float)new_height;

    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            float x = min(j * x_ratio, (float)(image.cols - 1)); // Clamp to the maximum valid index
            float y = min(i * y_ratio, (float)(image.rows - 1)); // Clamp to the maximum valid index

            resized_image.at<Vec3b>(i, j)[0] = bilinear_interpolation(image, x, y, 0);
            resized_image.at<Vec3b>(i, j)[1] = bilinear_interpolation(image, x, y, 1);
            resized_image.at<Vec3b>(i, j)[2] = bilinear_interpolation(image, x, y, 2);
        }
    }
    return resized_image;
}

Mat resize_EASU(const Mat& image, int new_width, int new_height) {
    Mat resized_image = Mat::zeros(new_height, new_width, CV_8UC3);
    float x_ratio = (float)image.cols / (float)new_width;
    float y_ratio = (float)image.rows / (float)new_height;

    for (int i = 0; i < new_height; i++) {
        for (int j = 0; j < new_width; j++) {
            float x = min(j * x_ratio, (float)(image.cols - 1)); // Clamp to the maximum valid index
            float y = min(i * y_ratio, (float)(image.rows - 1)); // Clamp to the maximum valid index

            if (i >= 4 && i < (new_height - 4) && j >= 4 && j < (new_height - 4)) {
                resized_image.at<Vec3b>(i, j) = EASU(image, x, y);
            }
            else {
                resized_image.at<Vec3b>(i, j)[0] = bilinear_interpolation(image, x, y, 0);
                resized_image.at<Vec3b>(i, j)[1] = bilinear_interpolation(image, x, y, 1);
                resized_image.at<Vec3b>(i, j)[2] = bilinear_interpolation(image, x, y, 2);
            }

        }
    }
    return resized_image;
}

Vec3b RCAS(Mat& img, int x, int y) {
    Vec3f pix(0.0, 0.0, 0.0);
    float bR = (float)img.at<Vec3f>(y - 1, x)[2];
    float bG = (float)img.at<Vec3f>(y - 1, x)[1];
    float bB = (float)img.at<Vec3f>(y - 1, x)[0];

    float dR = (float)img.at<Vec3f>(y, x - 1)[2];
    float dG = (float)img.at<Vec3f>(y, x - 1)[1];
    float dB = (float)img.at<Vec3f>(y, x - 1)[0];

    float eR = (float)img.at<Vec3f>(y, x)[2];
    float eG = (float)img.at<Vec3f>(y, x)[1];
    float eB = (float)img.at<Vec3f>(y, x)[0];

    float fR = (float)img.at<Vec3f>(y, x + 1)[2];
    float fG = (float)img.at<Vec3f>(y, x + 1)[1];
    float fB = (float)img.at<Vec3f>(y, x + 1)[0];

    float hR = (float)img.at<Vec3f>(y + 1, x)[2];
    float hG = (float)img.at<Vec3f>(y + 1, x)[1];
    float hB = (float)img.at<Vec3f>(y + 1, x)[0];

    float bL = bB * 0.5f + (bR * 0.5f + bG);
    float dL = dB * 0.5f + (dR * 0.5f + dG);
    float eL = eB * 0.5f + (eR * 0.5f + eG);
    float fL = fB * 0.5f + (fR * 0.5f + fG);
    float hL = hB * 0.5f + (hR * 0.5f + hG);
    float mn1L = std::min(AMin3F1(bL, dL, fL), hL);
    float mx1L = std::max(AMax3F1(bL, dL, fL), hL);
    Vec2f peakC(1.0f, -1.0f * 4.0f);

    float hitMinL = std::min(mn1L, eL) / (4.0f * mx1L + 0.00001f);
    float hitMaxL = (peakC[0] - std::max(mx1L, eL)) / (4.0f * mn1L + peakC[1]);

    float lobeL = std::max(-hitMinL, hitMaxL);
    float lobe = std::max(float(-FSR_RCAS_LIMIT), std::min(lobeL, 0.0f)) * exp2(-clamp(float(SHARPNESS), 0.0f, 2.0f));
    float nz = 0.25f * bL + 0.25f * dL + 0.25f * fL + 0.25f * hL - eL;
    nz = clamp(std::abs(nz) * APrxRcp(AMax3F1(AMax3F1(bL, dL, eL), fL, hL) - AMin3F1(AMin3F1(bL, dL, eL), fL, hL)), 0.0, 1.0);
    nz = -0.5f * nz + 1.0f;
    lobe *= nz;
    pix[2] = ((lobe * bR + lobe * dR + lobe * hR + lobe * fR + eR) / (4.0f * lobe + 1.0f));
    pix[1] = ((lobe * bG + lobe * dG + lobe * hG + lobe * fG + eG) / (4.0f * lobe + 1.0f));
    pix[0] = ((lobe * bB + lobe * dB + lobe * hB + lobe * fB + eB) / (4.0f * lobe + 1.0f));
    return pix;
}

void apply_filter(Mat& img) {
    cv::Mat result = img.clone();
    Mat img2;
    img.convertTo(img2, CV_32FC3);
    
        //Mat result = Mat::zeros(new_height, new_width, CV_8UC3);
        for (int y = 20; y < img.rows - 10; y++) {
            for (int x = 20; x < img.cols - 10; x++) {
                result.at<Vec3b>(y, x) = RCAS(img2, x, y);
            }
        }
        
        
    img = result;
}

int main() {
    //Loading image
    Mat image = imread("test_image.png", IMREAD_COLOR);

    if (image.empty()) {
        cout << "Could not open the image file: test.png" << endl;
        return -1;
    }

    //EASU
    Mat resized_image = resize_EASU(image, 512, 512);
    
    //RCAS
    apply_filter(resized_image);

    //Saving image
    imwrite("FSR.png", resized_image);
    return 0;
}