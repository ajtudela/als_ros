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

#ifndef __MRF_FAILURE_DETECTOR_H__
#define __MRF_FAILURE_DETECTOR_H__

#include <iostream>
#include <cmath>
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <visualization_msgs/Marker.h>

namespace als_ros {

enum MeasurementClass {
    ALIGNED = 0,
    MISALIGNED = 1,
    UNKNOWN = 2
};

class MRFFD {
public:
    MRFFD();
    ~MRFFD(){};

    inline void setMaxResidualError(double maxResidualError) { maxResidualError_ = maxResidualError; }
    inline void setNDMean(double NDMean) { NDMean_ = NDMean; }
    inline void setNDVariance(double NDVar) { NDVar_ = NDVar, NDNormConst_ = 1.0 / sqrt(2.0 * M_PI * NDVar_); }
    inline void setEDLambda(double EDLambda) { EDLambda_ = EDLambda; }
    inline void setResidualErrorReso(double residualErrorReso) { residualErrorReso_ = residualErrorReso; }
    inline void setMinValidResidualErrorNum(int minValidResidualErrorNum) { minValidResidualErrorsNum_ = minValidResidualErrorNum; }
    inline void setMaxLPBComputationNum(int maxLPBComputationNum) { maxLPBComputationNum_ = maxLPBComputationNum; }
    inline void setSamplingNum(int samplingNum) { samplingNum_ = samplingNum; }
    inline void setMisalignmentRatioThreshold(double misalignmentRatioThreshold) { misalignmentRatioThreshold_ = misalignmentRatioThreshold; }
    inline void setTransitionProbMat(std::vector<double> transitionProbMat) { transitionProbMat_ = transitionProbMat; }
    inline void setCanUpdateResidualErrors(bool canUpdateResidualErrors) { canUpdateResidualErrors_ = canUpdateResidualErrors; }

    inline double getFailureProbability(void) { return failureProbability_; }
    inline double getMeasurementClassProbabilities(int errorIndex, int measurementClass) { return measurementClassProbabilities_[errorIndex][measurementClass]; }
    inline std::vector<double> getMeasurementClassProbabilities(int errorIndex) { return measurementClassProbabilities_[errorIndex]; }
    inline double getFailureDetectionHz(void) { return failureDetectionHz_; }

    void predictFailureProbability(void);
    void publishROSMessages(void);
    void printFailureProbability(void);

private:
    // ros subscribers and publishers
    ros::NodeHandle nh_;
    std::string residualErrorsName_;
    ros::Subscriber residualErrorsSub_;
    std::string failureProbName_, alignedScanName_, misalignedScanName_, unknownScanName_;
    ros::Publisher failureProbPub_, alignedScanPub_, misalignedScanPub_, unknownScanPub_;
    bool publishClassifiedScans_;

    std::string failureProbabilityMarkerName_, markerFrame_;
    ros::Publisher failureProbabilityMarkerPub_;
    bool publishFailureProbabilityMarker_;

    // parametsrs
    double maxResidualError_;
    double NDMean_, NDVar_, NDNormConst_, EDLambda_;
    int minValidResidualErrorsNum_, maxResidualErrorsNum_;
    int maxLPBComputationNum_;
    int samplingNum_;
    double residualErrorReso_;
    double misalignmentRatioThreshold_, unknownRatioThreshold_;
    std::vector<double> transitionProbMat_;

    sensor_msgs::LaserScan residualErrors_;
    std::vector<double> usedResidualErrors_;
    std::vector<int> usedScanIndices_;
    bool canUpdateResidualErrors_, gotResidualErrors_;
    double failureDetectionHz_;

    // results
    std::vector<std::vector<double>> measurementClassProbabilities_;
    double failureProbability_;
    ros::Time failureProbabilityStamp_;

    void residualErrorsCB(const sensor_msgs::LaserScan::ConstPtr &msg);

    inline double calculateNormalDistribution(double e) {
        return (0.95 * (2.0 * NDNormConst_ * exp(-((e - NDMean_) * (e - NDMean_)) / (2.0 * NDVar_))) + 0.05 * (1.0 / maxResidualError_)) * residualErrorReso_;
    }

    inline double calculateExponentialDistribution(double e) {
        return (0.95 * (1.0 / (1.0 - exp(-EDLambda_ * maxResidualError_))) * EDLambda_ * exp(-EDLambda_ * e) + 0.05 * (1.0 / maxResidualError_)) * residualErrorReso_;
    }

    inline double calculateUniformDistribution(void) {
        return (1.0 / maxResidualError_) * residualErrorReso_;
    }

    inline double getSumOfVecotr(std::vector<double> vector) {
        double sum = 0.0;
        for (int i = 0; i < (int)vector.size(); i++)
            sum += vector[i];
        return sum;
    }

    inline std::vector<double> getHadamardProduct(std::vector<double> vector1, std::vector<double> vector2) {
        for (int i = 0; i < (int)vector1.size(); i++)
            vector1[i] *= vector2[i];
        return vector1;
    }

    inline std::vector<double> normalizeVector(std::vector<double> vector) {
        double sum = getSumOfVecotr(vector);
        for (int i = 0; i < (int)vector.size(); i++)
            vector[i] /= sum;
        return vector;
    }

    inline double getEuclideanNormOfDiffVectors(std::vector<double> vector1, std::vector<double> vector2) {
        double sum = 0.0;
        for (int i = 0; i < (int)vector1.size(); i++) {
            double diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    inline std::vector<double> calculateTransitionMessage(std::vector<double> probs) {
        std::vector<double> message(3);
        std::vector<double> tm = transitionProbMat_;
        message[ALIGNED] = tm[ALIGNED] * probs[ALIGNED] + tm[MISALIGNED] * probs[MISALIGNED] + tm[UNKNOWN] * probs[UNKNOWN];
        message[MISALIGNED] = tm[ALIGNED + 3] * probs[ALIGNED] + tm[MISALIGNED + 3] * probs[MISALIGNED] + tm[UNKNOWN + 3] * probs[UNKNOWN];
        message[UNKNOWN] = tm[ALIGNED + 6] * probs[ALIGNED] + tm[MISALIGNED + 6] * probs[MISALIGNED] + tm[UNKNOWN + 6] * probs[UNKNOWN];
        return message;
    }

    std::vector<std::vector<double>> getLikelihoodVectors(std::vector<double> validResidualErrors);
    std::vector<std::vector<double>> estimateMeasurementClassProbabilities(std::vector<std::vector<double>> likelihoodVectors);
    double predictFailureProbabilityBySampling(std::vector<std::vector<double>> measurementClassProbabilities);
    void setAllMeasurementClassProbabilities(std::vector<double> residualErrors, std::vector<std::vector<double>> measurementClassProbabilities);
    std::vector<int> getResidualErrorClasses(void);
}; // class MRFFD

} // namespace als_ros

#endif // __MRF_FAILURE_DETECTOR_H__
