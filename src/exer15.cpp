/* 
 * File:   exer15.cpp
 * Author: gregw
 *
 * Library code for exercises for chapter 15 of Learning OpenCV 3 - removing the background.
 * Moved this code from main.cpp to here to make testing it somewhat easier and cleaner.
 * 
 * Created on March 4, 2019, 1:16 PM
 */


#include "exer15.h"

using namespace std;


// Global storage.
// Float, 3-channel images.
cv::Mat image;
cv::Mat IavgF, IdiffF, IprevF, IhiF, IlowF;
cv::Mat tmp, tmp2;

// Float, 1-channel images
vector<cv::Mat> Igray(3);
vector<cv::Mat> Ilow(3);
vector<cv::Mat> Ihi(3);

// Byte, 1-channel image.
cv::Mat Imaskt;

// Count number of images learned, for averaging later.
float Icount;

// Used for accumulate* version.
cv::Mat sum, sqsum;
int image_count;

// Initialize all necessary intermediate images.
// I is just a sample image for allocation purposes - must have correct size.
// (passed in for sizing.)
void AllocateImages(const cv::Mat &I) {
    cout << "1" << endl;
    cv::Size sz = I.size();
    
    IavgF  = cv::Mat::zeros(sz, CV_32FC3);
    IdiffF = cv::Mat::zeros(sz, CV_32FC3);
    IprevF = cv::Mat::zeros(sz, CV_32FC3);
    IhiF   = cv::Mat::zeros(sz, CV_32FC3);
    IlowF  = cv::Mat::zeros(sz, CV_32FC3);
    Icount = 0.00001;       // Non-zero to protect against divide by zero.
    
    tmp    = cv::Mat::zeros(sz, CV_32FC3);
    tmp2   = cv::Mat::zeros(sz, CV_32FC3);
    Imaskt = cv::Mat(sz, CV_32FC1);
    
}


// Reset totals, between calls.
void ResetTotals() {
    Icount = 0.00001;
    sum = 0;
    sqsum = 0;
    image_count = 0.00001;
}


// Test functions, to set thresholds to known values.
void SetIlow(cv::Size matSize, int lowVal) {
    
    IlowF = cv::Mat::zeros(matSize, CV_32FC3);
    IlowF.setTo(lowVal);
    cv::split(IlowF, Ilow);
}


void SetIhi(cv::Size matSize, int highVal) {
    
    IhiF = cv::Mat::zeros(matSize, CV_32FC3);
    IhiF.setTo(highVal);
    cv::split(IhiF, Ihi);
}


// Learn the background statistics for an additional frame.
// I is a color sample of the background, 3-channel, 8U.
void accumulateBackground(cv::Mat &I) {
    cout << "2" << endl;
    static int first = 1;       // nb: not thread safe.
    I.convertTo(tmp, CV_32F);
    if( !first ) {
        IavgF += tmp;
        cv::absdiff( tmp, IprevF, tmp2);
        IdiffF += tmp2;
        Icount += 1.0;
    }
    first = 0;
    IprevF = tmp;
}


// Once have enough statistics, convert to model.
void createModelsFromStats() {
    cout << "3" << endl;
    IavgF *= (1.0/Icount);
    IdiffF *= (1.0/Icount);
    
    // Make sure diff is always something.
    IdiffF += cv::Scalar(1.0, 1.0, 1.0);
    setHighThreshold(7.0);
    setLowThreshold(6.0);
}


void setHighThreshold(float scale) {
    IhiF = IavgF + (IdiffF * scale);
    cv::split(IhiF, Ihi);
}


void setLowThreshold(float scale) {
    IlowF = IavgF - (IdiffF * scale);
    cv::split(IlowF, Ilow);
}


// Remove some noise by eroding the masks.
void erodeMask(cv::Mat &mask, cv::Mat &newMask) {
    cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
    cv::erode(mask, newMask, erodeKernel);
}


// Create a binary: 0,255 mask where 255 (white) means foreground pixel
// Pixel is considered foreground if all 3 channels are in range.
// I        Input image, 3-channel, 8U.
// Imask    Mask image to be created, 1-channel, 8U.
// erodeFlag Flag for whether to erode mask after creating, to reduce noise.
void backgroundDiff( cv::Mat &I, cv::Mat &Imask, bool erodeFlag) {
    //cout << "4" << endl;
    // Convert given image to float.
    I.convertTo(tmp, CV_32FC1);
    cv::split(tmp, Igray);
    
    // Channel 1. inRange returns 255 if inside range, 0 if outside.
    cv::inRange(Igray[0], Ilow[0], Ihi[0], Imask);
    //cout << "Igray 0: " << Igray[0] << endl;
    //cout << "Ilow 0: " << Ilow[0] << endl;
    //cout << "Ihi 0: " << Ihi[0] << endl;
    //cout << "Imask 0: " << Imask << endl;
    
    // Channel 2.
    cv::inRange(Igray[1], Ilow[1], Ihi[1], Imaskt);
    //cout << "Igray 1: " << Igray[1] << endl;
    //cout << "Imaskt 1: " << Imaskt << endl;
    Imask = cv::min(Imask, Imaskt);
    //cout << "Imask 1: " << Imask << endl;
    
    // Channel 3.
    cv::inRange(Igray[2], Ilow[2], Ihi[2], Imaskt);
    //cout << "Igray 2: " << Igray[2] << endl;
    //cout << "Imaskt 2: " << Imaskt << endl;
    Imask = cv::min(Imask, Imaskt);
    //cout << "Imask 2: " << Imask << endl;
    
    // Finally, invert the results, so foreground areas contain 255, background 0.
    Imask = 255 - Imask;
    
    // Clean up some noise by eroding.
    if (erodeFlag) {
        cv::Mat tmpMask;
        erodeMask(Imask, tmpMask);
        Imask = tmpMask.clone();
    }
}

void help(char **argv) {
    cout << "\n"
            << "Train a background model on incoming video, then run the model\n"
            << argv[0] << " avi_file\n"
            << endl;
}


// Create model, using simple sum.
void createModel_1(char **argv, cv::VideoCapture cap) {
    // First processing loop - training.
    bool first_frame = true;
    int frame_count = 0;
    while (1) {
        cap >> image;
        if (!image.data) {
            if (frame_count == 0) {
                // If weren't able to read any frames, exit.
                cout << "Unable to read any frames." << endl;
                exit(0);
            }
            else break;
        }
        ++frame_count;
        
        if( first_frame) {
            AllocateImages(image);
            first_frame = false;
        }

        accumulateBackground(image);
        
        cv::imshow(argv[1], image);
        if(cv::waitKey(7) == 0x20) break;
    }
    cout << "frame_count: " << frame_count << endl;

    // We have all our data, so create the models.
    createModelsFromStats();
    cv::destroyAllWindows();
    cout << "exam 7" << endl;
}


// Once have model, use it to highlight foreground in video.
void highlightForeground(char **argv) {

    cv::VideoCapture cap;
    cap.open(argv[1]);  // To test on the same video.
    //cap.open(argv[2]);  // To test on a separate video.
    
    // Second processing loop - testing.
    cv::Mat mask;
    while(1) {
        cap >> image;
        if (!image.data) exit(0);
        
        backgroundDiff(image, mask, true);
    //break;
        // A simple visualization is to write to the red channel.
        cv::split(image, Igray);
        Igray[2] = cv::max(mask, Igray[2]);
        cv::merge(Igray, image);
        
        cv::imshow(argv[1], image);
        if(cv::waitKey(7) == 0x20) break;
    }
}


// Using averages/absdiff to subtract background.
// A:   Found that this code flags everything as foreground - everything ends up red.
//      So, created modified version, to accept two files. First is of background only,
//      second includes moving objects. Thus, it trains on first video, the does diff on second.
//      But, still ended up with everything red. Problem turned out to be with initializing IdiffF -
//      I had typed 1,0, 1.0, 1.0, resulting in the second channel starting at 0!
//      Overall, seems a bit sensitive to changes - was catching reflections of cars in windows!
//      Increasing the high/low thresholds (i.e. to 11/10 from 7/6) reduced sensitivity, but just a bit.
void example15_2(int argc, char** argv) {

    if (argc < 2) {
        help(argv);
        return;
    }
    
    cv::namedWindow(argv[0], cv::WINDOW_AUTOSIZE);
    
    cv::VideoCapture cap;
    if((argc < 2) || !cap.open(argv[1])) {
        cout << "Couldn't open video file" << endl;
        help(argv);
        cap.open(0);
        return;
    }

    createModel_1(argv, cap);

    // Reset the video stream.
    cap.release();

    highlightForeground(argv);
    
    return;
}


// Calculated weighted mean and covariance. Stores covariance in sqsum, even 
// though it isn't actually the squared sum.
void accumulateMeans(cv::Mat &I) {

    if (sum.empty()) {
        sum = cv::Mat::zeros(I.size(), CV_32FC(I.channels()));
        sqsum = cv::Mat::zeros(I.size(), CV_32FC(I.channels()));
    }
    if (IprevF.empty()) {
        IprevF = cv::Mat::zeros(I.size(), CV_32F);
    }
    
    cv::accumulateWeighted(I, sum, 0.5);

    I.convertTo(tmp, CV_32F);
    cv::absdiff(tmp, IprevF, tmp2);
    cv::accumulateWeighted(tmp2, sqsum, 0.5);
    IprevF = tmp;

    ++image_count;
}


void setHighThreshold_2(float scale) {
    IhiF = sum + (sqsum * scale);
    cv::split(IhiF, Ihi);
}


void setLowThreshold_2(float scale) {
    IlowF = sum - (sqsum * scale);
    cv::split(IlowF, Ilow);
}


void createModelsFromStats_2() {
    
    // Make sure diff is always something.
    sqsum += cv::Scalar(1.0, 1.0, 1.0);

    // Clean up some of the noise in the mask
    setHighThreshold_2(12.0);
    setLowThreshold_2(11.0);
}


// Create model, using accumulateWeighted.
void createModel_2(char **argv, cv::VideoCapture cap) {
    // First processing loop - training.
    bool first_frame = true;
    int frame_count = 0;
    while (1) {
        cap >> image;
        if (!image.data) {
            if (frame_count == 0) {
                // If weren't able to read any frames, exit.
                cout << "Unable to read any frames." << endl;
                exit(0);
            }
            else break;
        }
        ++frame_count;
        
        if( first_frame) {
            AllocateImages(image);
            first_frame = false;
        }

        accumulateMeans(image);
        
        cv::imshow(argv[1], image);
        if(cv::waitKey(7) == 0x20) break;
    }
    cout << "frame_count: " << frame_count << endl;

    // We have all our data, so create the models.
    createModelsFromStats_2();
    cv::destroyAllWindows();
}


// Use averaging method of background subtraction, with accumulateWeighted() and the running average of 
// the absolute difference (absdiff()) as a proxy for the standard deviation.
// Starting with example 15.2 from book.
// A:   Doesn't look too different from example 15.2. Probably could simplify code a bit more, but it seems
//      to work as is.
void exer15_1(int argc, char **argv) {

    cv::namedWindow(argv[0], cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    if((argc < 2) || !cap.open(argv[1])) {
        cout << "Couldn't open video file" << endl;
        help(argv);
        cap.open(0);
        return;
    }

    // Train.
    createModel_2(argv, cap);

    // Reset the video stream.
    cap.release();

    // Run model against video.
    highlightForeground(argv);
    
    return;
}


// Try to highlight shadows with blue. In areas already masked (have mask value 255)as
// foreground, if red and green are below minimum, plus
// some threshold, but blue is above average value for given pixel, mark it as shadow.
void highlightShadows(cv::Mat mask, vector<cv::Mat> &imgChannels) {

    int thresh = -40;
    //int thresh = 125;
    int blueDiff, greenDiff, redDiff;
    float pixelsFlipped = 0;
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            if (mask.at<uchar>(row, col) == 255) {
                // If all values are below background range, and thus area is darker than it was, max the blue.
                blueDiff  = imgChannels[0].at<uchar>(row,col) - (Ilow[0].at<float>(row,col) + thresh);
                greenDiff = imgChannels[1].at<uchar>(row,col) - Ilow[1].at<float>(row,col);
                redDiff   = imgChannels[2].at<uchar>(row,col) - Ilow[2].at<float>(row,col);
                if (blueDiff < 0 && greenDiff < 0 && redDiff < 0) {
                    imgChannels[0].at<uchar>(row,col) = 255;
                    //imgChannels[1].at<uchar>(row,col) = 255;
                    //imgChannels[2].at<uchar>(row,col) = 255;
                    ++pixelsFlipped;
                    //cout << col << "," << row << ": " << blueDiff << "," << greenDiff << "," << redDiff;
                    //cout << "******" << endl;
                }
            }
        }
    }
    //cout << "Pixels changed: " << pixelsFlipped << endl;
}


static void onMouse_2( int event, int x, int y, int, void* param ) {
    if( event != cv::EVENT_LBUTTONDOWN ) return;
    
    auto ibgr = image.at<cv::Vec3b>(y, x);
    cout << "Clicked at: " << x << "," << y
            << "img b,g,r: " << (int) ibgr[0] << ", " << (int) ibgr[1] << ", " << (int) ibgr[2]
            << "Ilo b,g,r: " << Ilow[0].at<float>(y, x) << ", " << Ilow[1].at<float>(y, x)
            << ", " << Ilow[2].at<float>(y, x)
            << "Ihi b,g,r: " << Ihi[0].at<float>(y, x) << ", " << Ihi[1].at<float>(y, x)
            << ", " << Ihi[2].at<float>(y, x)
            << endl;
    
}

// Once have model, use it to highlight foreground in video. Modified version for
// exercise 15.2, removing shadows.
void highlightForeground_2(string filename) {

    cv::namedWindow(filename,  cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(filename, onMouse_2 );
    cv::VideoCapture cap;
    cap.open(filename);  // To test on the same video.

    cv::Mat erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
    
    // Second processing loop - testing.
    cv::Mat mask, origBlue;
    bool proceed = true;
    while(1) {
        if (proceed) {
            cap >> image;
            if (!image.data) break;

            backgroundDiff(image, mask, true);
            // A simple visualization is to write to the red channel.
            // Start by splitting image into individual arrays for each color.
            cv::split(image, Igray);

            cv::imshow("mask", mask);
            origBlue = Igray[0].clone();
            highlightShadows(mask, Igray);
    
            // Erode the blue channel to remove some more noise.
            cv::Mat tmpGray;
            cv::erode(Igray[0], tmpGray, erodeKernel);
            Igray[0] = tmpGray;
            //cv::imshow("Igray", Igray[0]);
            //cv::imshow("tmpGray", tmpGray);
            
            // Show original and new blue channels side by side.
            //cv::imshow("orig", origBlue);
            //cv::imshow("noShadow", Igray[0]);
            //cv::waitKey();

            cv::merge(Igray, image);

            cv::imshow(filename, image);
        }
        uchar lastKey = cv::waitKey(70);
        if (lastKey == 32) break;
        if (lastKey == 'p') proceed = false;
        if (lastKey == 'c') proceed = true;
    }
    
    // Display average image.
    //cv::Mat sumImg;
    //sum.convertTo(sumImg, CV_8U);
    //cv::imshow("sum", sum);
    //cv::imshow("sumImg", sumImg);
    //describe_mat(sum, "sum");
    //describe_mat(sumImg, "sumImg");
    cv::waitKey(0);
}


// Remove shadows from moving objects in image, so they don't appear as foreground objects.
// Use fact that they are darker (and bluer, when outdoors) than objects. For phase 1 am assuming that just want 
// to remove them as detected foreground objects. Then, maybe replace them with average value? How to avoid removing 
// black car, as well as its shadow?
// Approach: If marked as foreground, and overall darker than average for area, blue even further from average, mark
// it as shadow - a second mask. Use that to highlight area with blue, to show what was found.
// Starting with exer15_1, modifying highlightForeground to find shadows.
// A:   Result is moderately successful - some noise left, very difficult to differentiate between shadow and a dark car.
void exer15_2(int argc, char** argv) {

    cv::namedWindow(argv[0], cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    if((argc < 2) || !cap.open(argv[1])) {
        cout << "Couldn't open video file" << endl;
        help(argv);
        cap.open(0);
        return;
    }

    // Train.
    createModel_2(argv, cap);

    // Reset the video stream.
    cap.release();

    // Run model against video - either same as original, or separate if given two files.
    if (argc < 3) {
        highlightForeground_2(argv[1]);
    }
    else {
        highlightForeground_2(argv[2]);
    }
    
    return;
}


// Method to clean up a mask, taken from Example 15.5 in book.
void findConnectedComponents( cv::Mat &mask,
                            int poly1_hull0,
                            float perimScale,
                            vector<cv::Rect> &bbs,
                            vector<cv::Point> &centers,
                            float dp_epsilon_denominator,
                            int cvCloseIters) {

    // Clean up raw mask.
    cv::morphologyEx(mask, mask, cv::MorphTypes::MORPH_OPEN, cv::Mat(), cv::Point(-1,-1), cvCloseIters);
    cv::morphologyEx(mask, mask, cv::MorphTypes::MORPH_CLOSE, cv::Mat(), cv::Point(-1,-1), cvCloseIters);
    
    // Find contours only around bigger regions.
    vector<vector<cv::Point>> contours_all;
    vector<vector<cv::Point>> contours;     // Will contain just those that we wish to keep.
    cv::findContours(mask, contours_all, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    for( vector<vector<cv::Point>>::iterator c = contours_all.begin(); c != contours_all.end(); ++c) {
        // Get length of this contour.
        int len = cv::arcLength(*c, true);
        
        // Compare to perimeter of entire image.
        double q = (mask.rows + mask.cols) / dp_epsilon_denominator;
        if (len >= q ) {
            vector<cv::Point> c_new;
            if (poly1_hull0) {      // Caller wants polygons.
                cv::approxPolyDP(*c, c_new, len/20.0, true);
            }
            else {
                cv::convexHull(*c, c_new);
            }
            contours.push_back(c_new);
        }
    }
    
    const cv::Scalar CVX_WHITE = cv::Scalar(0xff, 0xff, 0xff);
    const cv::Scalar CVX_BLACK = cv::Scalar(0x00, 0x00, 0x00);
    
    // Calculate center of mass and/or bounding rectangles.
    int idx = 0;
    cv::Moments moments;
    cv::Mat scratch = mask.clone();
    for (vector<vector<cv::Point>>::iterator c = contours.begin(); c != contours.end(); ++c, ++idx) {
        cv::drawContours(scratch, contours, idx, CVX_WHITE, CV_FILLED);
        
        // Find the center of each contour.
        moments = cv::moments(scratch, true);
        cv::Point p;
        p.x = (int)(moments.m10 / moments.m00);
        p.y = (int)(moments.m01 / moments.m00);
        centers.push_back(p);
        bbs.push_back(cv::boundingRect(*c));
        scratch.setTo(0);
    }
    
    // Paint the found regions back into the image.
    mask.setTo(0);
    cv::drawContours(mask, contours, -1, CVX_WHITE);
}




// Once have model, use it to highlight foreground in video. Modified version for
// exercise 15.4, cleaning up mask before applying it.
// A:   Results. 0, 4, 20, 1. Lots of bad foreground objects found. Sometimes two cars combined
//              into one object. Sometimes one car consists of multiple objects.
//              1, 4, 20, 1. Fewer bad objects, but surprisingly small. Some cars not picked up - overall worse.
//              1, 2, 20, 1. Fewer bad objects, but still some surprisingly small. Fewer big objects as well - worse again.
//              0, 2, 20, 3. Many fewer bad objects, more good objects. Sometimes cars ghosted - contour still 
//              there, but car gone. Definitely better.
//              0, 2, 20, 5. Not much better - more cases of cars merging into one.
//              1, 2, 20, 3. Almost no noise, but cars weren't picked up very well - more pieces to each car.
//              0, 2, 40, 3. Only noise left reflections in windows, so not really noise. Cars look better - a bit less merging.
//              0, 3, 40, 3. Looks worse - like more combining and ghosting.
//              0, 2, 20, 3. Pretty similar to 0,2,20,3 - some combining, some ghosting, not really any noise - these appear
//              to be the two best combinations, for video traffic8.mov.
void highlightForeground_4(string filename) {

    cv::VideoCapture cap;
    cap.open(filename);  // To test on the same video.
    
    // Parameters for cleaning up mask.
    int poly = 0;        // poly1_hull0 - return polygons if 1, otherwise convex hulls;
    float perim = 2;     // perimScale. If perim of contour is less than (width+height entire image)/perimscale, remove it.
    float eps = 20;       // dp_epsilon_denominator.
    int closeIters = 3;  // Number of iterations of opening/closing.
    
    vector<cv::Rect> bbs;
    vector<cv::Point> centers;
    // Second processing loop - testing.
    cv::Mat mask;
    while(1) {
        cap >> image;
        if (!image.data) break;
        
        backgroundDiff(image, mask, true);
        findConnectedComponents(mask, poly, perim, bbs, centers, eps, closeIters);

        // Highlight foreground objects in red.
        cv::split(image, Igray);
        Igray[2] = cv::max(mask, Igray[2]);
        cv::merge(Igray, image);
        
        cv::imshow(filename, image);
        if(cv::waitKey(70) == 0x20) break;
    }
    
    cv::waitKey(0);
}


// Use background segmentation and image cleanup, try varying the following parameters
// to see result: poly1_hull0, DP_EPSILON_DENOMINATOR, perimScale, CVCLOSE_ITR.
// Start with 15_1, then call new method to clean up the mask before applying it.
void exer15_4(int argc, char** argv) {
 
    cv::namedWindow(argv[0], cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    if((argc < 2) || !cap.open(argv[1])) {
        cout << "Couldn't open video file" << endl;
        help(argv);
        cap.open(0);
        return;
    }

    // Train.
    createModel_2(argv, cap);

    // Reset the video stream.
    cap.release();

    // Run model against video.
    highlightForeground_4(argv[1]);
    
    return;

}


// Set max red value for all pixels in mask.
void highlightMask(cv::Mat &img, const cv::Mat mask) {

    cv::split(img, Igray);

    // Highlight foreground objects in red.
    Igray[2] = cv::max(mask, Igray[2]);
    cv::merge(Igray, img);
}


// Compare mog and mog2 for segmenting moving hand in front of tree - tree.avi.
// A:   Mog seems a lot better - less tricked by moving tree, moving tree masks less of hand,
//      when it appears, and left edge doesn't get flagged once hand appears.
void exer15_5(string videoName) {
    
    // open video, create output windows.
    string origWin = "original", mogWin = "MOG", mog2Win = "MOG2";
    cv::namedWindow(origWin, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(mogWin, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(mog2Win, cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    cap.open(videoName);
    
    // Create backgroundSubtractors, mog and mog2 versions.
    cv::Ptr<cv::bgsegm::BackgroundSubtractorMOG> mog = cv::bgsegm::createBackgroundSubtractorMOG(200, 5, 0.7, 15);
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 = cv::createBackgroundSubtractorMOG2(500, 16, true);

    // for each frame.
    cv::Mat srcImg, mogImg, mog2Img, mogMask, mog2Mask;
    int frameCount = 0;
    while(++frameCount < 200) {
        cap >> srcImg;
        if (!srcImg.data) break;

        // apply each subtractor to current frame.
        mog->apply(srcImg, mogMask, -1);
        mog2->apply(srcImg, mog2Mask, -1);
        //describe_mat(srcImg, "srcImg");
        //describe_mat(mogImg, "mogImg");
        //break;

        // Display original image, along with results from each mog.
        mogImg = srcImg.clone();
        mog2Img = srcImg.clone();
        highlightMask(mogImg, mogMask);
        highlightMask(mog2Img, mog2Mask);
        cv::imshow(origWin, srcImg);
        cv::imshow(mogWin, mogImg);
        cv::imshow(mog2Win, mog2Img);
        cv::waitKey(1000);
    }
    describe_mat(srcImg, "srcImg");
    describe_mat(mogMask, "mogMask");
    
    cap.release();
            
    return;
}


// Use bilateralFilter on video input before segmenting. Book says to use codebook
// background segmentation routine, but I'm going to use MOG, maybe one of the new ones as well,
// as I'm too lazy to type in the entire codebook code.
// Starting with exer15_5, then adding bilateralFilter, replacing mog2 with GSOC (picked at random from list in doc.)
// Adding filter may have improved gsoc a bit when no hand, but not mog, for tree video.
// For traffic8, gsoc did a much better job of finding the complete car, but suffered more from ghosting, and
// was slower to recognize when cars appeared. Used 20, 0.003, 0.01, 32, 0.01, 0.0022, 0.2, 0.1, 0.0004, 0.0008.
// When switched back to default values for GSOC, more noise, but picked up cars more quickly.
void exer15_6(string videoName) {
    // open video, create output windows.
    string origWin = "original", mogWin = "MOG", gsocWin = "GSOC";
    cv::namedWindow(origWin, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(mogWin, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(gsocWin, cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    cap.open(videoName);
    
    // Create backgroundSubtractors, mog and mog2 versions.
    cv::Ptr<cv::bgsegm::BackgroundSubtractorMOG> mog = cv::bgsegm::createBackgroundSubtractorMOG(200, 5, 0.7, 15);
    cv::Ptr<cv::bgsegm::BackgroundSubtractorGSOC> gsoc = cv::bgsegm::createBackgroundSubtractorGSOC(
            cv::bgsegm::LSBP_CAMERA_MOTION_COMPENSATION_NONE,
            20,         // Number of samples to maintain at each point in the frame.
            0.003f,     // Probability of replacing the old sample - how fast the model updates itself.
            0.01f,      // Probability of propagating to neighbors.
            32,         // How many positives the sample must get before it will be considered as a possible replacement.
            0.01f,      // Scale coefficient for threshold.
            0.0022f,    // Bias coefficient for threshold.
            0.1f,       // Blinking suppression decay factor.
            0.1f,       // Blinking suppression multiplier.
            0.0004f,    // Strength of the noise removal for background points.
            0.0008f);   // Strength of the noise removal for foreground points.
                        // Notes: 20, 0.03, 0.01, 32, 0.01, 0.0022, 0.1, 0.2, 0.0004, 0.0008 had a lot of noise in leaves
                        //          when using tree.avi, but hand was very clearly delineated.
                        //      Changing old sample prob from 0.03 to 0.003, 0.0003, didn't have much effect on noise.
                        //      Changing # positive samples before changing from 32 to 10 to 3 didn't have much effect on noise.
                        //      Changing noise remove strengths from 0.0004/0.0008 to 0.004/0.008 didn't have much effect either.
                        //          Changing further to 0.04/0.08 cleaned up noise a bit when there was no hand, but greatly
                        //          increased it when there was noise.

    // for each frame.
    cv::Mat srcImg, filtImg, mogImg, gsocImg, mogMask, gsocMask;
    int frameCount = 0;
    while(++frameCount < 200) {
        cap >> srcImg;
        if (!srcImg.data) break;

        // Apply bilateral filter to src image.
        cv::bilateralFilter(srcImg, filtImg, -1, 10, 10);
        
        // apply each subtractor to current frame.
        mog->apply(filtImg, mogMask, -1);
        gsoc->apply(filtImg, gsocMask, -1);
        //describe_mat(srcImg, "srcImg");
        //describe_mat(mogImg, "mogImg");
        //break;

        // Display original image, along with results from each mog.
        mogImg = srcImg.clone();
        gsocImg = srcImg.clone();
        highlightMask(mogImg, mogMask);
        highlightMask(gsocImg, gsocMask);
        cv::imshow(origWin, srcImg);
        cv::imshow(mogWin, mogImg);
        cv::imshow(gsocWin, gsocImg);
        cv::waitKey(100);
    }
    
    cap.release();
            
    return;
    
}


// Some test helpers. Should turn parts of this file into a class to clean this up.
cv::Mat& getSum() {
    return sum;
}

cv::Mat& getSqSum() {
    return sqsum;
}

int getImage_count() {
    return image_count;
}

