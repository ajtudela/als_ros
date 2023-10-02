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

#ifndef __MAE_CLASSIFIER_H__
#define __MAE_CLASSIFIER_H__

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <list>
#include <numeric>
#include <als_ros/Histogram.h>

namespace als_ros {

class MAEClassifier {
public:
    MAEClassifier();
    ~MAEClassifier();

    inline void setClassifierDir(std::string classifiersDir) { classifiersDir_ = classifiersDir; }
    inline void setMaxResidualError(double maxResidualError) { maxResidualError_ = maxResidualError; }
    inline void setMAEHistogramBinWidth(double maeHistogramBinWidth) { maeHistogramBinWidth_ = maeHistogramBinWidth; }

    inline double getFailureThreshold(void) { return failureThreshold_; }

    double getMAE(std::vector<double> residualErrors);
    void learnThreshold(std::vector<std::vector<double>> trainSuccessResidualErrors, std::vector<std::vector<double>> trainFailureResidualErrors);
    void writeClassifierParams(std::vector<std::vector<double>> testSuccessResidualErrors, std::vector<std::vector<double>> testFailureResidualErrors);
    void readClassifierParams(void);
    double calculateDecisionModel(double mae, double *reliability);
    void writeDecisionLikelihoods(void);

private:
    std::string classifiersDir_;
    double maxResidualError_;

    double failureThreshold_;
    double successMAEMean_, successMAEStd_;
    double failureMAEMean_, failureMAEStd_;

    int truePositiveNum_, falsePositiveNum_, trueNegativeNum_, falseNegativeNum_;
    double dTruePositive_, dFalsePositive_, dTrueNegative_, dFalseNegative_;

    double maeHistogramBinWidth_;
    Histogram positiveMAEHistogram_, negativeMAEHistogram_;
    Histogram truePositiveMAEHistogram_, trueNegativeMAEHistogram_;
    Histogram falsePositiveMAEHistogram_, falseNegativeMAEHistogram_;

    template <template<class T, class Allocator = std::allocator<T>> class Container> double getMean(Container<double> &x) {
        return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    }

    template <template<class T, class Allocator = std::allocator<T>> class Container> double getVar(Container<double> &x) {
        double size = x.size();
        double mean = getMean(x);
        return (std::inner_product(x.begin(), x.end(), x.begin(), 0.0) - mean * mean * size) / (size - 1.0);
    }

    template <template<class T, class Allocator = std::allocator<T>> class Container> double getStd(Container<double> &x) {
        return std::sqrt(getVar(x));
    }

    std::vector<double> readMAEs(std::string filePath);
}; // class MAEClassifier

} // namespace als_ros

#endif // __MAE_CLASSIFIER_H__
