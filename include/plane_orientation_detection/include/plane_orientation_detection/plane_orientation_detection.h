

#ifndef PLANE_ORIENTATION_DETECTION
#define PLANE_ORIENTATION_DETECTION

#include <plane_detection_utils/plane_detection_utils.h>

#include <vector>


std::vector<double> find_ab(const LineSegment& li, const LineSegment& lj, double f_val);
std::vector<double> findDominantPlane(std::vector<uint8_t>& inliners, const std::vector<LineSegment>& lines,
                                      const std::vector<std::vector<int>>& adjacency_graph, double f, double t, int N);
std::vector<PlaneInfo> getPlanes(const std::vector<LineSegment>& lines, double f, double t, int N,
                                 double relative_dist_threshold, double angle_threshold,
                                 double plane_relative_threshold, double plane_absolute_threshold);

#endif  // PLANE_ORIENTATION_DETECTION