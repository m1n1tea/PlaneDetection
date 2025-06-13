

#ifndef PLANE_LABELING
#define PLANE_LABELING

#include <plane_detection_utils/plane_detection_utils.h>

#include <opencv2/core.hpp>
#include <set>
#include <vector>

void distributeLinesBeetweenPlanes(std::vector<PlaneInfo> &planes, const std::vector<LineSegment> &lines, double f,
                                   double t, double relative_dist_threshold, double angle_threshold);

std::vector<cv::Point2f> getRect(const LineSegment &li, const LineSegment &lj, double f, double a, double b);

float findGoodnessScore(const std::vector<cv::Point2f> &polygon, int plane,
                        const std::vector<std::pair<cv::Point2f, int>> &intersection_points);

float getMinX(const std::vector<cv::Point2f> &polygon);
float getMaxX(const std::vector<cv::Point2f> &polygon);
bool areConvexPolygonsIntersect(const std::vector<cv::Point2f> &poly1, const std::vector<cv::Point2f> &poly2);
std::vector<std::set<int>> findConflictGraph(const std::vector<std::vector<cv::Point2f>> &polygons,
                                             const std::vector<int> &polygons_plane, int planes_count);

void drawContour4(cv::Mat &image, std::vector<cv::Point2f> vertices2f, cv::Scalar color);

std::vector<cv::Mat> findPlaneRegions(const std::vector<PlaneInfo> &planes, const std::vector<LineSegment> &lines,
                                      double f, cv::Size img_size);

#endif  // PLANE_LABELING