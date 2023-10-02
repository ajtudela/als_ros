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

#ifndef __SLAMER_H__
#define __SLAMER_H__

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <als_ros/MCL.h>
#include <als_ros/ISM.h>
#include <als_ros/LineObjectRecognition.h>

namespace als_ros {

class SLAMER : public MCL {
public:
    SLAMER(std::string ismYamlFile);
    ~SLAMER(){};
    void calculateLikelihoodsBySLAMER(void);
    void recognizeObjectsWithMapAssist(void);
    void publishSLAMERROSMessages(void);

private:
    ros::NodeHandle nh_;
    ros::Publisher ismPointsPub_, coloredScanPointsPub_, lineObjectsPub_, spatialLineObjectsPub_;
    std::string ismPointsName_, coloredScanPointsName_, lineObjectsName_, spatialLineObjectsName_;
    bool publishISMPoints_, publishColoredScanPoints_;

    ISM ism_;
    std::vector<std::string> objectNames_;
    double mapResolution_;

    double normConstHit_, denomHit_, zHit_, measurementModelRandom_;

    sensor_msgs::LaserScan scan_;
    LineObjectRecognition recog_;
    std::vector<LineObject> lineObjects_, spatialLineObjects_;
    std::vector<std::vector<double>> lineObjectsProbs_, spatialLineObjectsProbs_;

    inline double clipProb(double p) {
        if (p < 10.0e-15)
            return 10.0e-15;
        else if (p > 1.0 - 10.0e-15)
            return 1.0 - 10.0e-15;
        return p;
    }

    inline double calculateDirichletDistribution(std::vector<double> probs, std::vector<double> coefs) {
        double prod1 = 1.0, prod2 = 1.0, sum = 0.0;
        for (int i = 0; i < (int)probs.size(); ++i) {
            prod1 *= std::tgamma(coefs[i]);
            sum += coefs[i];
            prod2 *= pow(probs[i], coefs[i] - 1.0);
        }
        double beta = prod1 / std::tgamma(sum);
        double p = prod2 / beta;
        return clipProb(p);
    }

    double calculateLineObjectClassificationModel(std::vector<double> classificationProbs, int objectID);
    double calculateSpatialLineObjectClassificationModel(std::vector<double> classificationProbs, int objectID);
    double calculateLikelihoodFieldModel(double dist);
    std::vector<double> getLineObjectPrior(Pose sensorPose, LineObject lineObject);
    std::vector<double> getSpatialLineObjectPrior(Pose sensorPose, LineObject spatialLineObject);
}; // class SLAMER : public MCL

} // namespace als_ros

#endif // __SLAMER_H__