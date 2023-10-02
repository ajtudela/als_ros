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

#ifndef __ISM_H__
#define __ISM_H__

#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <als_ros/Pose.h>

namespace als_ros {

class ISM {
public:
    ISM();
    ~ISM();
    void readISM(std::string ismYamlFile);

    inline std::vector<std::string> getObjectNames(void) { return objectNames_; }
    inline double getMapResolution(void) { return mapResolution_; }

    inline sensor_msgs::PointCloud2 getISMPointsAsPC2(void) {
        sensor_msgs::PointCloud2 ismPointsMsg;
        pcl::toROSMsg(*ismPoints_.get(), ismPointsMsg);
        return ismPointsMsg;
    }

    inline float getDistance(double x, double y, std::string object) {
        int u, v;
        xy2uv<double>(x, y, &u, &v);
        if (u < 0 || mapWidth_ <= u || v < 0 || mapHeight_ <= v)
            return -1.0f;

        for (int i = 0; i < (int)objectNames_.size(); ++i) {
            if (objectNames_[i] == object)
                return distMaps_[i].at<float>(v, u);
        }
        return -1.0f;
    }

private:
    std::vector<std::string> objectNames_;
    std::vector<std::vector<int>> objectColors_;

    int mapWidth_, mapHeight_;
    double mapResolution_;
    Pose mapOrigin_;

    std::vector<cv::Mat> distMaps_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ismPoints_;

    template <typename T>
    inline void xy2uv(T x, T y, int *u, int *v) {
        *u = (int)((x - mapOrigin_.getX()) / mapResolution_);
        *v = mapHeight_ - 1 - (int)((y - mapOrigin_.getY()) / mapResolution_);
    }

    template <typename T>
    inline void uv2xy(int u, int v, T *x, T *y) {
        *x = (T)u * mapResolution_ + mapOrigin_.getX();
        *y = (T)(mapHeight_ - 1 - v) * mapResolution_ + mapOrigin_.getY();
    }
}; // class ISM

} // namespace als_ros

#endif // __ISM_H__