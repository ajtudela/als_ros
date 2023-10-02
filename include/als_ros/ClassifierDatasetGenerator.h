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

#ifndef __CLASSIFIER_DATASET_GENERATOR_H__
#define __CLASSIFIER_DATASET_GENERATOR_H__

#include <tf/tf.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <opencv2/opencv.hpp>
#include <als_ros/Pose.h>

namespace als_ros {

class Obstacle {
public:
    double x_, y_, s_;
    // x: x position
    // y: y position
    // s: size

    Obstacle(void):
        x_(0.0), y_(0.0), s_(1.0) {}

    Obstacle(double x, double y, double s):
        x_(x), y_(y), s_(s) {}
}; // class Obstacle

class ClassifierDatasetGenerator {
public:
    ClassifierDatasetGenerator();
    ~ClassifierDatasetGenerator();
    void datasetGenerationInit();
    void generateDataset();

    inline void setTrainDirs(std::vector<std::string> trainDirs) {
        trainDirs_ = trainDirs;
    }

    inline void setTestDirs(std::vector<std::string> testDirs) {
        testDirs_ = testDirs;
    }

    void readTrainDataset(std::vector<Pose> &gtPoses, std::vector<Pose> &successPoses, std::vector<Pose> &failurePoses,
        std::vector<sensor_msgs::LaserScan> &scans, std::vector<std::vector<double>> &successResidualErrors, std::vector<std::vector<double>> &failureResidualErrors);
    void readTestDataset(std::vector<Pose> &gtPoses, std::vector<Pose> &successPoses, std::vector<Pose> &failurePoses,
        std::vector<sensor_msgs::LaserScan> &scans, std::vector<std::vector<double>> &successResidualErrors, std::vector<std::vector<double>> &failureResidualErrors);

private:
    ros::NodeHandle nh_;

    std::string mapName_;
    ros::Subscriber mapSub_;

    int generateSampleNum_;
    std::string saveDir_;
    std::vector<std::string> trainDirs_, testDirs_;

    nav_msgs::OccupancyGrid map_;
    cv::Mat distMap_;
    double mapResolution_;
    Pose mapOrigin_;
    int mapWidth_, mapHeight_;
    bool gotMap_;
    double freeSpaceMinX_, freeSpaceMaxX_, freeSpaceMinY_, freeSpaceMaxY_;

    int obstaclesNum_;

    double angleMin_, angleMax_, angleIncrement_, rangeMin_, rangeMax_, scanAngleNoise_, scanRangeNoise_;
    double validScanRateTH_;

    double failurePositionalErrorTH_, failureAngularErrorTH_;
    double positionalErrorMax_, angularErrorMax_;

    inline double nrand(double n) {
        return (n * sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * rand() / RAND_MAX));
    }

    inline double urand(double min, double max) {
        return ((max - min)  * (double)rand() / RAND_MAX + min);
    }

    inline bool onMap(int u, int v) {
        if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_)
            return true;
        else
            return false;
    }

    inline void xy2uv(double x, double y, int *u, int *v) {
        double dx = x - mapOrigin_.getX();
        double dy = y - mapOrigin_.getY();
        double yaw = -mapOrigin_.getYaw();
        double xx = dx * cos(yaw) - dy * sin(yaw);
        double yy = dx * sin(yaw) + dy * cos(yaw);
        *u = (int)(xx / mapResolution_);
        *v = (int)(yy / mapResolution_);
    }

    inline void uv2xy(int u, int v, double *x, double *y) {
        double xx = (double)u * mapResolution_;
        double yy = (double)v * mapResolution_;
        double yaw = mapOrigin_.getYaw();
        double dx = xx * cos(yaw) - yy * sin(yaw);
        double dy = xx * sin(yaw) + yy * cos(yaw);
        *x = dx + mapOrigin_.getX();
        *y = dy + mapOrigin_.getY();
    }

    inline int xy2node(double x, double y) {
        int u, v;
        xy2uv(x, y, &u, &v);
        if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_)
            return v * mapWidth_ + u;
        else
            return -1;
    }

    void mapCB(const nav_msgs::OccupancyGrid::ConstPtr &msg);
    std::vector<Obstacle> generateObstacles();
    nav_msgs::OccupancyGrid buildSimulationMap(std::vector<Obstacle> obstacles);
    void generatePoses(Pose &gtPose, Pose &successPose, Pose &failurePose);
    sensor_msgs::LaserScan simulateScan(Pose gtPose, nav_msgs::OccupancyGrid simMap);
    bool isValidScan(sensor_msgs::LaserScan scan);
    std::vector<double> getResidualErrors(Pose pose, sensor_msgs::LaserScan scan);
    void saveData(Pose gtPose, Pose successPose, Pose failurePose, sensor_msgs::LaserScan scan, std::vector<double> successResidualErrors, std::vector<double> failureResidualErrors);
    void readDataset(std::vector<std::string> dirs, std::vector<Pose> &gtPoses, std::vector<Pose> &successPoses, std::vector<Pose> &failurePoses,
        std::vector<sensor_msgs::LaserScan> &scans, std::vector<std::vector<double>> &successResidualErrors, std::vector<std::vector<double>> &failureResidualErrors);
}; // class ClassifierDatasetGenerator

} // namespace als_ros

#endif // __CLASSIFIER_DATASET_GENERATOR_H__
