#include <als_ros/LineObjectRecognition.h>

namespace als_ros {

void LineObjectRecognition::init(double scanRangeMax, double scanMapReso, double scanOrientationHistReso, double minScanOrientationProb,
        double minSpatialObjectLineLength, double maxSpatialObjectLineLength, double mergePointsDist)
{
    scanRangeMax_ = scanRangeMax;
    scanMapReso_ = scanMapReso;
    scanOrientationHistReso_ = scanOrientationHistReso;
    minScanOrientationProb_ = minScanOrientationProb;
    minSpatialObjectLineLength_ = minSpatialObjectLineLength;
    maxSpatialObjectLineLength_ = maxSpatialObjectLineLength;
    mergePointsDist_ = mergePointsDist;

    scanOrientationHistBinMax_ = (int)(M_PI / scanOrientationHistReso_);
}

void LineObjectRecognition::recognizeLineObjects(sensor_msgs::LaserScan scan, std::vector<LineObject> &lineObjects, std::vector<std::vector<double>> &lineObjectsProbs,
        std::vector<LineObject> &spatialLineObjects, std::vector<std::vector<double>> &spatialLineObjectsProbs)
{
    cv::Mat scanImg = makeScanImage(scan);
    cv::Mat expScanImg = expandScanImg(scanImg);
    std::vector<cv::Vec4i> scanLines = HoughTransform(scanImg);
    std::vector<double> scanOriHist = getScanOrientationHistogram(scanLines);
    scanImgPoints_ = getScanImgPoints(scanImg);
    lineObjects = makeLineObjects(expScanImg, scanLines, scanOriHist);
    spatialLineObjects = makeSpatialLineObjects(expScanImg, scanImgPoints_, scanOriHist);

    lineObjects = mergeLineObjects(lineObjects);
    spatialLineObjects = mergeLineObjects(spatialLineObjects);
    spatialLineObjects = checkScanPassability(spatialLineObjects, scanImgPoints_);

    lineObjectsProbs.resize((int)lineObjects.size());
    for (int i = 0; i < (int)lineObjects.size(); ++i)
        lineObjectsProbs[i] = classifyLineObject(lineObjects[i]);

    spatialLineObjectsProbs.resize((int)spatialLineObjects.size());
    for (int i = 0; i < (int)spatialLineObjects.size(); ++i)
        spatialLineObjectsProbs[i] = classifySpatialLineObject(spatialLineObjects[i]);

    // plotScan(scanImgPoints_, "/tmp/scan_points.txt");
    // plotScan(, "/tmp/vscan_points.txt");
    // printLineObjects(lineObjects, "line objects");
    // printLineObjects(spatialLineObjects, "spatial line objects");
    // writeLineObjects(lineObjects, "/tmp/line_objects.txt");
    // writeLineObjects(spatialLineObjects, "/tmp/spatial_line_objects.txt");
}

sensor_msgs::PointCloud2 LineObjectRecognition::getColoredScanImgPoints(std::vector<LineObject> lineObjects, std::vector<std::vector<double>> lineObjectsProbs) {
    std::vector<cv::Mat> lineImgs(4);
    for (int i = 0; i < (int)lineImgs.size(); ++i)
        lineImgs[i] = cv::Mat::ones(scanMapHeight_, scanMapWidth_, CV_8UC1);

    for (int i = 0; i < (int)lineObjects.size(); ++i) {
        double max;
        int maxIdx;
        getMaxProbAndIdx(lineObjectsProbs[i], max, maxIdx);
        als_ros::Point p1 = lineObjects[i].getP1();
        als_ros::Point p2 = lineObjects[i].getP2();
        int u1, v1, u2, v2;
        xy2uv(p1.getX(), p1.getY(), &u1, &v1);
        xy2uv(p2.getX(), p2.getY(), &u2, &v2);
        if (max < 0.5)
            continue;

        if (maxIdx == (int)LineObjectLabel::CLOSE_DOOR)
            cv::line(lineImgs[0], cv::Point(u1, v1), cv::Point(u2, v2), cv::Scalar(0), 1, cv::LINE_AA);
        else if (maxIdx == (int)LineObjectLabel::CLOSE_GLASS_DOOR)
            cv::line(lineImgs[1], cv::Point(u1, v1), cv::Point(u2, v2), cv::Scalar(0), 1, cv::LINE_AA);
        else if (maxIdx == (int)LineObjectLabel::FENCE)
            cv::line(lineImgs[2], cv::Point(u1, v1), cv::Point(u2, v2), cv::Scalar(0), 1, cv::LINE_AA);
        else
            cv::line(lineImgs[3], cv::Point(u1, v1), cv::Point(u2, v2), cv::Scalar(0), 1, cv::LINE_AA);
    }

    std::vector<cv::Mat> distMaps((int)lineImgs.size());
    for (int i = 0; i < (int)distMaps.size(); ++i) {
        distMaps[i] = cv::Mat(scanMapHeight_, scanMapWidth_, CV_32FC1);
        cv::distanceTransform(lineImgs[i], distMaps[i], cv::DIST_L2, 5);
        for (int v = 0; v < scanMapHeight_; v++) {
            for (int u = 0; u < scanMapWidth_; u++) {
                float d = distMaps[i].at<float>(v, u) * (float)scanMapReso_;
                distMaps[i].at<float>(v, u) = d;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredPoints(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (int i = 0; i < (int)scanImgPoints_.size(); ++i) {
        als_ros::Point sip = scanImgPoints_[i];
        int u, v;
        xy2uv(sip.getX(), sip.getY(), &u, &v);
        float min = distMaps[0].at<float>(v, u);
        int minIdx = 0;
        for (int j = 1; j < (int)distMaps.size(); ++j) {
            float d = distMaps[j].at<float>(v, u);
            if (d < min) {
                min = d;
                minIdx = j;
            }
        }

        float r, g, b;
        if (minIdx == 0)
            getObjectColor((int)LineObjectLabel::CLOSE_DOOR, r, g, b);
        else if (minIdx == 1)
            getObjectColor((int)LineObjectLabel::CLOSE_GLASS_DOOR, r, g, b);
        else if (minIdx == 2)
            getObjectColor((int)LineObjectLabel::FENCE, r, g, b);
        else
            getObjectColor((int)LineObjectLabel::OTHERS, r, g, b);

        pcl::PointXYZRGB p;
        p.x = sip.getX();
        p.y = sip.getY();
        p.z = 0.0;
        p.r = (int)(r * 255.0f);
        p.g = (int)(g * 255.0f);
        p.b = (int)(b * 255.0f);
        coloredPoints->points.push_back(p);
    }
    coloredPoints->width = coloredPoints->points.size();
    coloredPoints->height = 1;

    sensor_msgs::PointCloud2 coloredPointsMsg;
    pcl::toROSMsg(*coloredPoints.get(), coloredPointsMsg);
    return coloredPointsMsg;
}

cv::Mat LineObjectRecognition::makeScanImage(sensor_msgs::LaserScan scan) {
    scanMapWidth_ = (int)(2.0 * scanRangeMax_ / scanMapReso_);
    scanMapHeight_ = (int)(2.0 * scanRangeMax_ / scanMapReso_);
    scanAngleReso_ = scan.angle_increment;
    cv::Mat scanImg = cv::Mat::zeros(scanMapHeight_, scanMapWidth_, CV_8U);
    for (int i = 0; i < (int)scan.ranges.size(); ++i) {
        double r = scan.ranges[i];
        if (r < scan.range_min || scanRangeMax_ < r)
            continue;
        double t = scan.angle_min + (double)i * scan.angle_increment;
        double x = r * cos(t);
        double y = r * sin(t);
        int u, v;
        xy2uv(x, y, &u, &v);
        if (u < 0 || scanMapWidth_ <= u || v < 0 || scanMapHeight_ <= v)
            continue;
        scanImg.at<uchar>(v, u) = 255;
    }
/*
    cv::namedWindow("Image", 1);
    cv::imshow("Image", scanImg);
    cv::waitKey(30);
*/
    return scanImg;
}

cv::Mat LineObjectRecognition::expandScanImg(cv::Mat scanImg) {
    cv::Mat kernel; // 3 x 3
    int iteration = 1;
    cv::Mat expScanImg;
    cv::dilate(scanImg, expScanImg, kernel, cv::Point(-1, -1), iteration);
/*
    cv::namedWindow("Image", 1);
    cv::imshow("Image", expScanImg);
    cv::waitKey(30);
*/
    return expScanImg;
}

std::vector<als_ros::Point> LineObjectRecognition::getScanImgPoints(cv::Mat scanImg) {
    std::vector<als_ros::Point> scanImgPoints;
    for (int u = 0; u < scanImg.cols; ++u) {
        for (int v = 0; v < scanImg.rows; ++v) {
            if (scanImg.at<uchar>(v, u) == 255) {
                double x, y;
                uv2xy(u, v, &x, &y);
                scanImgPoints.push_back(Point(x, y));
            }
        }
    }
    return scanImgPoints;
}

std::vector<cv::Vec4i> LineObjectRecognition::HoughTransform(cv::Mat scanImg) {
    std::vector<cv::Vec4i> scanLines;
    // void HoughLinesP(Mat& image, vector<Vec4i>& lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0)
    cv::HoughLinesP(scanImg, scanLines, 1.0, CV_PI / 180.0, 5, 5.0, 10.0);

/*
    for(int i = 0; i < (int)scanLines.size(); ++i) {
        cv::Vec4i l = scanLines[i];
        cv::line(scanImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(100), 1, cv::LINE_AA);
    }
    cv::namedWindow("Image", 1);
    cv::imshow("Image", scanImg);
    cv::waitKey(30);
*/
    return scanLines;
}

std::vector<double> LineObjectRecognition::getScanOrientationHistogram(std::vector<cv::Vec4i> scanLines) {
    std::vector<double> hist(scanOrientationHistBinMax_);
    hist.resize(scanOrientationHistBinMax_, 0.0);
    int scanLineSize = (int)scanLines.size();
    if (scanLineSize == 0)
        return hist;

    for(int i = 0; i < scanLineSize; ++i) {
        cv::Vec4i l = scanLines[i];
        double x1, y1, x2, y2;
        uv2xy(l[0], l[1], &x1, &y1);
        uv2xy(l[2], l[3], &x2, &y2);
        double dx = x2 - x1;
        double dy = y2 - y1;
        double t = atan2(dy, dx);
        if (t < 0.0)
            t += M_PI;
        int b = (int)(t / scanOrientationHistReso_);
        if (b < 0)
            b = 0;
        else if (b >= scanOrientationHistBinMax_)
            b = scanOrientationHistBinMax_ - 1;
        hist[b] += 1.0;
    }

    for (int i = 0; i < scanOrientationHistBinMax_; ++i) {
        hist[i] /= (double)scanLineSize;
        // printf("i = %d, rad = %lf, val = %lf (%d)\n", i, (double)i * scanOrientationHistReso_, hist[i], scanLineSize);
    }
    return hist;
}

std::vector<LineObject> LineObjectRecognition::makeLineObjects(cv::Mat scanImg, std::vector<cv::Vec4i> scanLines, std::vector<double> scanOriHist) {
    std::vector<LineObject> lineObjects;
    for (int i = 0; i < (int)scanLines.size(); ++i) {
        cv::Vec4i l = scanLines[i];
        double x1, y1, x2, y2;
        uv2xy(l[0], l[1], &x1, &y1);
        uv2xy(l[2], l[3], &x2, &y2);
        double dx = x2 - x1;
        double dy = y2 - y1;
        double t = atan2(dy, dx);
        if (t < 0.0)
            t += M_PI;
        int b = (int)(t / scanOrientationHistReso_);
        if (b < 0)
            b = 0;
        else if (b >= scanOrientationHistBinMax_)
            b = scanOrientationHistBinMax_ - 1;
        if (scanOriHist[b] < minScanOrientationProb_)
            continue;
        double dl = sqrt(dx * dx + dy * dy);
        double ddx = scanMapReso_ * cos(t);
        double ddy = scanMapReso_ * sin(t);
        double x = x1;
        double y = y1;
        int scanPointsNum = 0;
        for (double l = 0.0; l <= dl; l += scanMapReso_) {
            int u, v;
            xy2uv(x, y, &u, &v);
            if (u < 0 || scanMapWidth_ <= u || v < 0 || scanMapHeight_ <= v)
                continue;
            if (scanImg.at<uchar>(v, u) > 250)
                scanPointsNum++;
            x += ddx;
            y += ddy;
        }
        LineObject line(als_ros::Point(x1, y1), als_ros::Point(x2, y2), scanPointsNum);
        lineObjects.push_back(line);
    }
    return lineObjects;
}

std::vector<double> LineObjectRecognition::classifyLineObject(LineObject lineObject) {
    als_ros::Point p1 = lineObject.getP1();
    als_ros::Point p2 = lineObject.getP2();
    int scanPointsNum = lineObject.getScanPointsNum();
    double dx = p2.getX() - p1.getX();
    double dy = p2.getY() - p1.getY();
    double t1 = atan2(p1.getY(), p1.getX());
    double t2 = atan2(p2.getY(), p2.getX());
    double dt = t2 - t1;
    while (dt > M_PI)
        dt -= 2.0 * M_PI;
    while (dt < M_PI)
        dt += 2.0 * M_PI;
    int lasersNum = (int)(dt / scanAngleReso_);
    int expMaxScanPointsNum = (int)(sqrt(dx * dx + dy * dy) / scanMapReso_);
    int expScanPointsNum;
    if (expMaxScanPointsNum > lasersNum)
        expScanPointsNum = lasersNum;
    else
        expScanPointsNum = expMaxScanPointsNum;
    double pointsRate = (double)scanPointsNum / (double)expScanPointsNum;
    if (pointsRate > 1.0 - 10.0e-6)
        pointsRate = 1.0 - 10.0e-6;

    std::vector<double> lineObjectProb((int)LineObjectLabel::OTHERS + 1);
    lineObjectProb[(int)LineObjectLabel::CLOSE_DOOR] = pointsRate;
    lineObjectProb[(int)LineObjectLabel::OPEN_DOOR] = 0.0;
    lineObjectProb[(int)LineObjectLabel::OPEN_GLASS_DOOR] = 0.0;
    lineObjectProb[(int)LineObjectLabel::CLOSE_GLASS_DOOR] = 0.5;
    if (pointsRate > 0.5)
        lineObjectProb[(int)LineObjectLabel::FENCE] = 0.5;
    else
        lineObjectProb[(int)LineObjectLabel::FENCE] = 1.0;
    lineObjectProb[(int)LineObjectLabel::NO_ENTRY] = 0.0;
    lineObjectProb[(int)LineObjectLabel::FREE_SPACE] = 0.0;
    lineObjectProb[(int)LineObjectLabel::OTHERS] = 1.0;
    normalizeProb(lineObjectProb);

    return lineObjectProb;
}

std::vector<LineObject> LineObjectRecognition::makeSpatialLineObjects(cv::Mat scanImg, std::vector<als_ros::Point> scanImgPoints, std::vector<double> scanOriHist) {
    std::vector<LineObject> spatialLineObjects;
    for (int i = 0; i < (int)scanImgPoints.size(); ++i) {
        for (int j = 0; j < (int)scanImgPoints.size(); ++j) {
            if (i == j)
                continue;
            double dx = scanImgPoints[j].getX() - scanImgPoints[i].getX();
            double dy = scanImgPoints[j].getY() - scanImgPoints[i].getY();
            double dl = sqrt(dx * dx + dy * dy);
            if (dl < minSpatialObjectLineLength_ || maxSpatialObjectLineLength_ < dl)
                continue;
            double t = atan2(dy, dx);
            double tt = t;
            if (tt < 0.0)
                tt += M_PI;
            int b = (int)(tt / scanOrientationHistReso_);
            if (b < 0)
                b = 0;
            if (b >= scanOrientationHistBinMax_)
                b = scanOrientationHistBinMax_ - 1;
            if (scanOriHist[b] < minScanOrientationProb_)
                continue;
            double ddx = scanMapReso_ * cos(t);
            double ddy = scanMapReso_ * sin(t);
            double x = scanImgPoints[i].getX();
            double y = scanImgPoints[i].getY();
            int scanPointsNum = 0;
            for (double l = scanMapReso_; l <= dl - scanMapReso_; l += scanMapReso_) {
                x += ddx;
                y += ddy;
                int u, v;
                xy2uv(x, y, &u, &v);
                if (scanImg.at<uchar>(v, u) > 250)
                    scanPointsNum++;
            }
            int expScanPointsNum = (int)(dl / scanMapReso_);
            double pointsRate = (double)scanPointsNum / (double)expScanPointsNum;
            if (pointsRate > 0.5)
                continue;
            LineObject line(scanImgPoints[i], scanImgPoints[j], scanPointsNum);
            spatialLineObjects.push_back(line);
        }
    }
    return spatialLineObjects;
}

std::vector<LineObject> LineObjectRecognition::checkScanPassability(std::vector<LineObject> spatialLineObjects, std::vector<als_ros::Point> scanImgPoints) {
    std::vector<double> scanAngles((int)scanImgPoints.size());
    for (int i = 0; i < (int)scanImgPoints.size(); ++i)
        scanAngles[i] = atan2(scanImgPoints[i].getY(), scanImgPoints[i].getX());

    std::vector<LineObject> passabileLines;
    for (int i = 0; i < (int)spatialLineObjects.size(); ++i) {
        als_ros::Point p1 = spatialLineObjects[i].getP1();
        als_ros::Point p2 = spatialLineObjects[i].getP2();
        double t1 = atan2(p1.getY(), p1.getX());
        double t2 = atan2(p2.getY(), p2.getX());
        double tmin, tmax;
        if (t1 < t2)
            tmin = t1, tmax = t2;
        else
            tmin = t2, tmax = t1;

        double passability = 0.5;
        double unpassibility = 1.0 - passability;
        for (int j = 0; j < (int)scanAngles.size(); ++j) {
            if (t1 <= scanAngles[j] &&  scanAngles[j] <= t2) {
                bool crossLines = checkLineCrossing(als_ros::Point(0.0, 0.0), scanImgPoints[j], p1, p2);
                if (crossLines) {
                    double d = compDistFromPointToLine(p1, p2, scanImgPoints[j]);
                    if (d > 0.5) {
                        passability *= 0.9;
                        unpassibility *= 0.1;
                    } else {
                        passability *= 0.1;
                        unpassibility *= 0.9;
                    }
                } else {
                    passability *= 0.1;
                    unpassibility *= 0.9;
                }
                double sum = passability + unpassibility;
                passability /= sum;
                unpassibility /= sum;
            }
        }
        if (passability > 0.9)
            passabileLines.push_back(spatialLineObjects[i]);
    }
    return passabileLines;
}

std::vector<double> LineObjectRecognition::classifySpatialLineObject(LineObject spatiallineObject) {
    als_ros::Point p1 = spatiallineObject.getP1();
    als_ros::Point p2 = spatiallineObject.getP2();
    int scanPointsNum = spatiallineObject.getScanPointsNum();
    double dx = p2.getX() - p1.getX();
    double dy = p2.getY() - p1.getY();
    double t1 = atan2(p1.getY(), p1.getX());
    double t2 = atan2(p2.getY(), p2.getX());
    double dt = t2 - t1;
    while (dt > M_PI)
        dt -= 2.0 * M_PI;
    while (dt < M_PI)
        dt += 2.0 * M_PI;
    int lasersNum = (int)(dt / scanAngleReso_);
    int expMaxScanPointsNum = (int)(sqrt(dx * dx + dy * dy) / scanMapReso_);
    int expScanPointsNum;
    if (expMaxScanPointsNum > lasersNum)
        expScanPointsNum = lasersNum;
    else
        expScanPointsNum = expMaxScanPointsNum;
    double pointsRate = (double)scanPointsNum / (double)expScanPointsNum;

    std::vector<double> spatialLineObjectProb((int)LineObjectLabel::OTHERS + 1);
    spatialLineObjectProb[(int)LineObjectLabel::CLOSE_DOOR] = 0.0;
    spatialLineObjectProb[(int)LineObjectLabel::OPEN_DOOR] = 1.0 - pointsRate;
    spatialLineObjectProb[(int)LineObjectLabel::OPEN_GLASS_DOOR] = 1.0 - pointsRate;
    spatialLineObjectProb[(int)LineObjectLabel::CLOSE_GLASS_DOOR] = 1.0 - pointsRate;
    if (pointsRate == 0.0)
        spatialLineObjectProb[(int)LineObjectLabel::FENCE] = 0.0;
    else
        spatialLineObjectProb[(int)LineObjectLabel::FENCE] = 1.0;
    spatialLineObjectProb[(int)LineObjectLabel::NO_ENTRY] = 1.0 - pointsRate;
    spatialLineObjectProb[(int)LineObjectLabel::FREE_SPACE] = 0.5;
    spatialLineObjectProb[(int)LineObjectLabel::OTHERS] = 1.0;
    normalizeProb(spatialLineObjectProb);

    return spatialLineObjectProb;
}

std::vector<LineObject> LineObjectRecognition::mergeLineObjects(std::vector<LineObject> lineObjects) {
    std::vector<LineObject> mergedLineObjects;
    for (;;) {
        mergedLineObjects.push_back(lineObjects[0]);
        lineObjects.erase(lineObjects.begin() + 0);
        int targetIdx = (int)mergedLineObjects.size() - 1;
        als_ros::Point p11 = mergedLineObjects[targetIdx].getP1();
        als_ros::Point p12 = mergedLineObjects[targetIdx].getP2();
        int scanPointsNum1 = mergedLineObjects[targetIdx].getScanPointsNum();
        for (int i = 0; i < (int)lineObjects.size(); ++i) {
            als_ros::Point p21 = lineObjects[i].getP1();
            als_ros::Point p22 = lineObjects[i].getP2();
            double dx1 = p21.getX() - p11.getX();
            double dy1 = p21.getY() - p11.getY();
            double dx2 = p22.getX() - p12.getX();
            double dy2 = p22.getY() - p12.getY();
            double dl1 = sqrt(dx1 * dx1 + dy1 * dy1);
            double dl2 = sqrt(dx2 * dx2 + dy2 * dy2);
            if (dl1 < mergePointsDist_ && dl2 < mergePointsDist_) {
                double x1 = (p21.getX() + p11.getX()) / 2.0;
                double y1 = (p21.getY() + p11.getY()) / 2.0;
                double x2 = (p22.getX() + p12.getX()) / 2.0;
                double y2 = (p22.getY() + p12.getY()) / 2.0;
                int scanPointsNum = (scanPointsNum1 + lineObjects[i].getScanPointsNum()) / 2;
                mergedLineObjects[targetIdx].setLineObject(als_ros::Point(x1, y1), als_ros::Point(x2, y2), scanPointsNum);
                p11 = mergedLineObjects[targetIdx].getP1();
                p12 = mergedLineObjects[targetIdx].getP2();
                scanPointsNum1 = mergedLineObjects[targetIdx].getScanPointsNum();
                lineObjects.erase(lineObjects.begin() + i);
                i--;
            }
        }
        if ((int)lineObjects.size() == 0)
            break;
    }
    return mergedLineObjects;
}

void LineObjectRecognition::plotScan(sensor_msgs::LaserScan scan, std::string file) {
    FILE *fp = fopen(file.c_str(), "w");
    for (int i = 0; i < (int)scan.ranges.size(); ++i) {
        double r = scan.ranges[i];
        if (r < scan.range_min || scanRangeMax_ < r)
            continue;
        double t = scan.angle_min + (double)i * scan.angle_increment;
        double x = r * cos(t);
        double y = r * sin(t);
        fprintf(fp, "%lf %lf\n", x, y);
    }
    fclose(fp);
}

void LineObjectRecognition::plotScan(std::vector<als_ros::Point> scanPoints, std::string file) {
    FILE *fp = fopen(file.c_str(), "w");
    for (int i = 0; i < (int)scanPoints.size(); ++i)
        fprintf(fp, "%lf %lf\n", scanPoints[i].getX(), scanPoints[i].getY());
    fclose(fp);
}

void LineObjectRecognition::printLineObjects(std::vector<LineObject> lineObjects, std::string type) {
    for (int i = 0; i < (int)lineObjects.size(); ++i) {
        als_ros::Point p1 = lineObjects[i].getP1();
        als_ros::Point p2 = lineObjects[i].getP2();
        int scanPointsNum = lineObjects[i].getScanPointsNum();
        printf("type = %s, x1 = %lf, y1 = %lf, x2 = %lf, y2 = %lf, scanPointsNum = %d\n",
            type.c_str(), p1.getX(), p1.getY(), p2.getX(), p2.getY(), scanPointsNum);
    }
    printf("\n");
}

void LineObjectRecognition::writeLineObjects(std::vector<LineObject> lineObjects, std::string fname) {
    FILE *fp = fopen(fname.c_str(), "w");
    for (int i = 0; i < (int)lineObjects.size(); ++i) {
        als_ros::Point p1 = lineObjects[i].getP1();
        als_ros::Point p2 = lineObjects[i].getP2();
        fprintf(fp, "%lf %lf\n", p1.getX(), p1.getY());
        fprintf(fp, "%lf %lf\n", p2.getX(), p2.getY());
        fprintf(fp, "\n");
    }
    fclose(fp);
}

}  // namespace als_ros