/****************************************************************************
 * als_ros: An Advanced Localization System for ROS use with 2D LiDAR
 * Copyright (C) 2022 Naoki Akai
 *
 * Licensed under the Apache License, Version 2.0 (the “License”);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @author Naoki Akai
 ****************************************************************************/

#ifndef __LINE_OBJECT_RECOGNITION_H__
#define __LINE_OBJECT_RECOGNITION_H__

#include <string>
#include <vector>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
#include <als_ros/Point.h>

namespace als_ros {

enum class LineObjectLabel {
    CLOSE_DOOR = 0,
    OPEN_DOOR = 1,
    OPEN_GLASS_DOOR = 2,
    CLOSE_GLASS_DOOR = 3,
    FENCE = 4,
    NO_ENTRY = 5,
    FREE_SPACE = 6,
    OTHERS = 7
};

class LineObject {
private:
    als_ros::Point p1_, p2_;
    int scanPointsNum_;

public:
    LineObject(void):
        p1_(als_ros::Point(0.0, 0.0)),
        p2_(als_ros::Point(0.0, 0.0)),
        scanPointsNum_(0) {}

    LineObject(double x1, double y1, double x2, double y2, int scanPointsNum):
        p1_(als_ros::Point(x1, y1)),
        p2_(als_ros::Point(x2, y2)),
        scanPointsNum_(scanPointsNum) {}

    LineObject(als_ros::Point p1, als_ros::Point p2, int scanPointsNum):
        p1_(p1),
        p2_(p2),
        scanPointsNum_(scanPointsNum) {}

    inline als_ros::Point getP1(void) { return p1_; }
    inline als_ros::Point getP2(void) { return p2_; }
    inline int getScanPointsNum(void) { return scanPointsNum_; }

    inline void setLineObject(als_ros::Point p1, als_ros::Point p2, int scanPointsNum) {
        p1_ = p1, p2_ = p2, scanPointsNum_= scanPointsNum;
    }
}; // class LineObject

class LineObjectRecognition {
public:
    LineObjectRecognition() {}
    ~LineObjectRecognition() {}

    void init(double scanRangeMax, double scanMapReso, double scanOrientationHistReso, double minScanOrientationProb,
        double minSpatialObjectLineLength, double maxSpatialObjectLineLength, double mergePointsDist);

    inline void normalizeProb(std::vector<double> &prob) {
        double sum = 0.0;
        for (int i = 0; i < (int)prob.size(); ++i)
            sum += prob[i];
        for (int i = 0; i < (int)prob.size(); ++i)
            prob[i] /= sum;
    }

    inline void getMaxProbAndIdx(std::vector<double> prob, double &maxProb, int &maxIdx) {
        maxProb = prob[0];
        maxIdx = 0;
        for (int i = 1; i < (int)prob.size(); ++i) {
            if (maxProb < prob[i]) {
                maxProb = prob[i];
                maxIdx = i;
            }
        }
    }

    inline void getObjectColor(int objectID, float &r, float &g, float &b) {
        if (objectID == (int)LineObjectLabel::CLOSE_DOOR)
            r = 167.0f, g = 87.0f, b = 168.0f;	
        else if (objectID == (int)LineObjectLabel::OPEN_DOOR)
            r = 0.0f, g = 204.0f, b = 102.0f;
        else if (objectID == (int)LineObjectLabel::OPEN_GLASS_DOOR)
            r = 196.0f, g = 184.0f, b = 155.0f;
        else if (objectID == (int)LineObjectLabel::CLOSE_GLASS_DOOR)
            r = 116.0f, g = 69.0f, b = 170.0f;
        else if (objectID == (int)LineObjectLabel::FENCE)
            r = 216.0f, g = 96.0f, b = 17.0f;
        else if (objectID == (int)LineObjectLabel::NO_ENTRY)
            r = 255.0f, g = 0.0f, b = 0.0f;
        else if (objectID == (int)LineObjectLabel::FREE_SPACE)
            r = 255.0f, g = 255.0f, b = 255.0f;
        else if (objectID == (int)LineObjectLabel::OTHERS)
            r = 0.0, g = 0.0f, b = 0.0f;
        else
            r = 255.0, g = 255.0, b = 255.0;
        r /= 255.0f;
        g /= 255.0f;
        b /= 255.0f;
    }

    void recognizeLineObjects(sensor_msgs::LaserScan scan, std::vector<LineObject> &lineObjects, std::vector<std::vector<double>> &lineObjectsProbs,
        std::vector<LineObject> &spatialLineObjects, std::vector<std::vector<double>> &spatialLineObjectsProbs);
    sensor_msgs::PointCloud2 getColoredScanImgPoints(std::vector<LineObject> lineObjects, std::vector<std::vector<double>> lineObjectsProbs);

private:
    double scanRangeMax_, scanMapReso_;
    double scanOrientationHistReso_, minScanOrientationProb_;
    double minSpatialObjectLineLength_, maxSpatialObjectLineLength_;
    double mergePointsDist_;

    std::vector<als_ros::Point> scanImgPoints_;
    int scanOrientationHistBinMax_;
    int scanMapWidth_, scanMapHeight_;
    double scanAngleReso_;

    inline void xy2uv(double x, double y, int *u, int *v) {
        *u = (int)(x / scanMapReso_) + scanMapWidth_ / 2;
        *v = scanMapHeight_ / 2 - (int)(y / scanMapReso_);
    }

    inline void uv2xy(int u, int v, double *x, double *y) {
        *x = (double)(u - scanMapWidth_ / 2) * scanMapReso_;
        *y = (double)(scanMapHeight_ / 2 - v) * scanMapReso_;
    }

    cv::Mat makeScanImage(sensor_msgs::LaserScan scan);
    cv::Mat expandScanImg(cv::Mat scanImg);
    std::vector<als_ros::Point> getScanImgPoints(cv::Mat scanImg);
    std::vector<cv::Vec4i> HoughTransform(cv::Mat scanImg);
    std::vector<double> getScanOrientationHistogram(std::vector<cv::Vec4i> scanLines);
    std::vector<LineObject> makeLineObjects(cv::Mat scanImg, std::vector<cv::Vec4i> scanLines, std::vector<double> scanOriHist);
    std::vector<double> classifyLineObject(LineObject lineObject);
    std::vector<LineObject> makeSpatialLineObjects(cv::Mat scanImg, std::vector<als_ros::Point> scanImgPoints, std::vector<double> scanOriHist);

    inline bool checkLineCrossing(als_ros::Point a, als_ros::Point b, als_ros::Point c, als_ros::Point d) {
        double s, t;
        s = (a.getX() - b.getX()) * (c.getY() - a.getY()) - (a.getY() - b.getY()) * (c.getX() - a.getX());
        t = (a.getX() - b.getX()) * (d.getY() - a.getY()) - (a.getY() - b.getY()) * (d.getX() - a.getX());
        if (s * t > 0)
            return false;
        s = (c.getX() - d.getX()) * (a.getY() - c.getY()) - (c.getY() - d.getY()) * (a.getX() - c.getX());
        t = (c.getX() - d.getX()) * (b.getY() - c.getY()) - (c.getY() - d.getY()) * (b.getX() - c.getX());
        if (s * t > 0)
            return false;
        return true;
    }

    inline double compDistFromPointToLine(als_ros::Point l1, als_ros::Point l2, als_ros::Point p) {
        double t = atan2(l2.getY() - l1.getY(), l2.getX() - l1.getX());
        double a, b, c;
        if (fabs(t) <= M_PI / 4.0 || 3.0 * M_PI / 4.0 <= fabs(t)) {
            a = (l2.getY() - l1.getY()) / (l2.getX() - l1.getX());
            b = -1.0;
            c = -a + l1.getY();
        } else {
            a = 1.0;
            b = -(l2.getX() - l1.getX()) / (l2.getY() - l1.getY());
            c = -b - l1.getX();
        }
        return fabs(a * p.getX() + b * p.getY() + c) / sqrt(a * a + b * b);
    }

    std::vector<LineObject> checkScanPassability(std::vector<LineObject> spatialLineObjects, std::vector<als_ros::Point> scanImgPoints);
    std::vector<double> classifySpatialLineObject(LineObject spatiallineObject);
    std::vector<LineObject> mergeLineObjects(std::vector<LineObject> lineObjects);
    void plotScan(sensor_msgs::LaserScan scan, std::string file);
    void plotScan(std::vector<als_ros::Point> scanPoints, std::string file);
    void printLineObjects(std::vector<LineObject> lineObjects, std::string type);
    void writeLineObjects(std::vector<LineObject> lineObjects, std::string fname);
}; // class LineObjectRecognition

} // namespace als_ros

#endif // __LINE_OBJECT_RECOGNITION_H__