#ifndef FOCAL_LENGTH_ESTIMATION
#define FOCAL_LENGTH_ESTIMATION

#include <plane_detection_utils/plane_detection_utils.h>

std::vector<cv::Point3d> findVanishingPoints(std::vector<LineSegment> &lines);
std::vector<std::vector<uint32_t>> findOrthMat(const std::vector<std::vector<int>> &adjacency_graph,
                                               const std::vector<LineSegment> &lines,
                                               const std::vector<cv::Point3d> &vanishing_points,
                                               double vanishing_points_relative_threshold,
                                               double vanishing_points_absolute_threshold);
double estimateFocalLength(const std::vector<cv::Point3d> &vanishing_points,
                           const std::vector<std::vector<uint32_t>> &orth_mat);
double findFocalLength(const std::vector<LineSegment> &lines_for_vanishing_points, double relative_dist_threshold,
                       double angle_threshold, double vanishing_points_relative_threshold,
                       double vanishing_points_absolute_threshold);

#endif  // FOCAL_LENGTH_ESTIMATION