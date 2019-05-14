/* 
 * File:   newsimpletest.cpp
 * Author: gregw
 *
 * Created on April 11, 2019, 11:21 AM
 */

#include <stdlib.h>
#include <iostream>

#include "gtest/gtest.h"

#include "../src/exer15.h"

using namespace std;

/*
 * GoogleTest Test Suite
 */

int compareMats(cv::Mat&, cv::Mat&);
void setValInPos(cv::Mat&, int, int, int);
void testBackgroundDiff(cv::Mat&, cv::Mat&);


TEST(TestExer15_1, testAccumulateMeans_zerogiveszero) {
    cv::Mat img = cv::Mat::zeros(3, 3, CV_8U);
    ResetTotals();
    accumulateMeans(img);
    cv::Mat expSum = cv::Mat::zeros(3, 3, CV_32F);
    expSum.setTo(0);
    int numDiffs = compareMats(getSum(), expSum);
    EXPECT_EQ(numDiffs, 0);
}


// Since accumulateMeans uses an alpha of 0.5, when add first image of all ones to it, expect
// sums to end up being 0.5
TEST(TestExer15_1, testAccumulateMeans_onegiveshalf) {
    cv::Mat img = cv::Mat::ones(3, 3, CV_8U);
    ResetTotals();
    accumulateMeans(img);
    cv::Mat expSum = cv::Mat::ones(3, 3, CV_32F);
    expSum.setTo(0.5);
    int numDiffs = compareMats(getSum(), expSum);
    EXPECT_EQ(numDiffs, 0);
}


// Adding two images, ones and twelves, expect 6.25 as result.
TEST(TestExer15_1, testAccumulateMeans_twoImages) {
    cv::Mat img = cv::Mat::ones(3, 3, CV_8U);
    ResetTotals();
    accumulateMeans(img);
    img.setTo(12);
    accumulateMeans(img);
    cv::Mat expSum = cv::Mat::ones(3, 3, CV_32F);
    expSum.setTo(6.25);
    int numDiffs = compareMats(getSum(), expSum);
    EXPECT_EQ(numDiffs, 0);
    EXPECT_EQ(getImage_count(), 2);
}


// BackgroundDiff should return an 8bit mask with background pixels = 0,
// foreground = 255. Background is defined as pixels having all three channels
// within range defined by Ilow and Ihigh. These tests put same value in all three
// channels of the test matrix, as simple tests.
TEST(TestExer15_2, testBackgroundDiff_singleChannel1) {
    // Order of data seems to be bgr pixel 1, bgr pixel 2, etc.
    uchar data[9] = {   0,   0,   0, 
                       10,  10,  10, 
                      210, 210, 210};
    cv::Mat img = cv::Mat(1, 3, CV_8UC3, data);
    
    uchar expData[3] = {255, 0, 0};
    cv::Mat expMask = cv::Mat(1, 3, CV_8U, expData);
    testBackgroundDiff(img, expMask);
}

TEST(TestExer15_2, testBackgroundDiff_singleChannel2) {
    uchar data[9]  = {   11,  11,  11,
                         50,  50,  50,
                        250, 250, 250};
    cv::Mat img = cv::Mat(1, 3, CV_8UC3, data);
    
    uchar expData[3] = {0, 0, 255};
    cv::Mat expMask = cv::Mat(1, 3, CV_8U, expData);
    testBackgroundDiff(img, expMask);
}

TEST(TestExer15_2, testBackgroundDiff_threeChannel1) {
    uchar data[12] = {  9,  9,  9, 
                       50,  9,  9, 
                       50, 50,  9,
                       50, 50, 50};
    cv::Mat img = cv::Mat(1, 4, CV_8UC3, data);
    
    uchar expData[4] = {255, 255, 255, 0};
    cv::Mat expMask = cv::Mat(1, 4, CV_8U, expData);
    testBackgroundDiff(img, expMask);
}


// This function should set blue channel of image to 255 in foreground areas
// where current image is darker and more blue than the background. Function accepts
// a mask (255=foreground) and an image, use Ihi as upper bound on background.
TEST(TestExer15_2, testHighlightShadows) {

    uchar imgData[12] = {    10,  10,  10,
                             40,  70,  70,
                            210, 210, 210,
                             40,  70,  70 };
    cv::Mat img = cv::Mat(1, 4, CV_8UC3, imgData);
    
    uchar maskData[4] = {0, 255, 255, 0};
    cv::Mat mask = cv::Mat(1, 4, CV_8U, maskData);
    
    SetIlow(img.size(), 100);

    vector<cv::Mat> imgChannels(3);
    cv::split(img, imgChannels);
    
    highlightShadows(mask, imgChannels);
    
    uchar expBlueData[4] = {10, 255, 210, 40};
    cv::Mat expBlue = cv::Mat(1, 4, CV_8U, expBlueData);
    int numDiffs = compareMats(imgChannels[0], expBlue);
    EXPECT_EQ(numDiffs, 0);
    
    if (numDiffs > 0) {
        cout << "img[0]: " << imgChannels[0] << endl
                << "exp: " << expBlue << endl;
    }

}

int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    
}


// Return number of elements which are not same in both matrices.
int compareMats(cv::Mat &mat1, cv::Mat &mat2) {
    cv::Mat diffs = cv::Mat::zeros(mat1.size(), mat1.type());
    describe_mat(mat1, "mat1");
    describe_mat(mat2, "mat2");
    describe_mat(diffs, "diffs");
    cv::absdiff(mat1, mat2, diffs);
    
    return cv::countNonZero(diffs);
}

// Helper function, to put given val in given position of given mat, in all 3 channels.
void setValInPos(cv::Mat &mat1, int row, int col, int val) {
 
    mat1.at<uchar>(row, col, 0) = val;
    mat1.at<uchar>(row, col, 1) = val;
    mat1.at<uchar>(row, col, 2) = val;
}


void testBackgroundDiff(cv::Mat &img, cv::Mat &expMask) {
    SetIlow(img.size(), 10);
    SetIhi(img.size(), 210);

    // Don't want to erode mask, as then harder to check result.
    cv::Mat actMask;
    backgroundDiff(img, actMask, false);
    
    int numDiffs = compareMats(actMask, expMask);
    EXPECT_EQ(numDiffs, 0);
    
    if (numDiffs > 0) {
        cout << "img: " << img << endl;
        cout    << "expected: " << expMask << endl
                << "actual:   " << actMask << endl;
    }
}
