

#ifndef PLANE_DETECTION_UTILS
#define PLANE_DETECTION_UTILS

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/optim.hpp>
#include <vector>

struct LineSegment
{
    LineSegment(cv::Point2d start, cv::Point2d end);
    LineSegment(double x1, double y1, double x2, double y2);
    LineSegment(cv::Vec4f vec = {});

    cv::Vec3d getLineHomogeneousCoordinates() const;
    cv::Point2d getVector() const;

    // lets pretend that start!=end is always true
    cv::Point2d start;
    cv::Point2d end;

    unsigned int vp_id;
};

struct PlaneInfo
{
    PlaneInfo() {}
    std::vector<std::vector<int>> adjacency_graph;
    double a;
    double b;
};

cv::Point2d getIntersection(const LineSegment &lhs, const LineSegment &rhs);
bool areAdjacent(const LineSegment &lhs, const LineSegment &rhs, double relative_dist_threshold,
                 double angle_threshold);

std::vector<std::vector<int>> findAdjacencyGraph(const std::vector<LineSegment> &lines, double relative_dist_threshold,
                                                 double angle_threshold);
void remove_lines(std::vector<LineSegment> &lines, std::vector<std::vector<int>> &adjacency_graph,
                  const std::vector<uint8_t> &mask, bool remove_zeros);

std::vector<LineSegment> detectLines(cv::Mat img);

class OrthoganalityMetricFunction : public cv::MinProblemSolver::Function
{
   public:
    OrthoganalityMetricFunction(const LineSegment &li, const LineSegment &lj, double f_val);
    int getDims() const override;
    double calc(const double *x) const override;

    static cv::Matx33d transformationMat(double a, double b, double f);
    static cv::Matx33d transformationMatInvT(double a, double b, double f);
    static cv::Matx33d transformationMatInv(double a, double b, double f);
    static cv::Matx33d xRotationMat(double a);
    static cv::Matx33d yRotationMat(double b);
    static cv::Matx33d calibrationMat(double f);
    static cv::Matx33d calibrationMatInv(double f);

   private:
    cv::Vec3d li;
    cv::Vec3d lj;
    double f_val;
};

#endif  // PLANE_DETECTION_plane_detection_utils