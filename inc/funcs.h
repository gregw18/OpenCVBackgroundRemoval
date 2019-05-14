/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   funcs.h
 * Author: gregw
 *
 * Created on February 23, 2019, 10:39 AM
 */

#ifndef FUNCS_H
#define FUNCS_H

#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat get_small_image(const std::string, cv::ImreadModes = cv::ImreadModes::IMREAD_COLOR, bool = true);
void describe_mat(cv::Mat, std::string);
void getStillsFromVideo(std::string);


#endif /* FUNCS_H */

