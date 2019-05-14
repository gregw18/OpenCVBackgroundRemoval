/* 
 * File:   exer15.h
 * Author: gregw
 *
 * Created on April 11, 2019, 11:26 AM
 */

#ifndef EXER15_H
#define EXER15_H

#include <cstdlib>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>
#include "../inc/funcs.h"

void example15_2(int, char**);

void exer15_1(int, char**);
void exer15_2(int, char**);
void exer15_4(int, char**);
void exer15_5(std::string);
void exer15_6(std::string);

void setHighThreshold(float);
void setLowThreshold(float);
void help(char**);

void AllocateImages(const cv::Mat&);
void accumulateBackground(cv::Mat&);
void createModelsFromStats();
void setHighThreshold(float);
void setLowThreshold(float);
void backgroundDiff( cv::Mat&, cv::Mat&, bool);
void createModel_1(char*, cv::VideoCapture);
void highlightForeground(char **);
void accumulateMeans(cv::Mat&);
void setHighThreshold_2(float);
void setLowThreshold_2(float);
void createModelsFromStats_2();
void createModel_2(char**, cv::VideoCapture);
void highlightShadows(cv::Mat, std::vector<cv::Mat>&);
void highlightForeground_2(std::string);
void findConnectedComponents( cv::Mat&, int, float, std::vector<cv::Rect>&, std::vector<cv::Point>&, float, int);
void highlightForeground_4(std::string);
void highlightMask(cv::Mat&, const cv::Mat);

cv::Mat& getSum();
cv::Mat& getSqSum();
int getImage_count();

// Used for testing.
void SetIlow(cv::Size, int);
void SetIhi(cv::Size, int);
void ResetTotals();


#endif /* EXER15_H */

