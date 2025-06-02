#include <opencv2/core.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <casadi/casadi.hpp>


struct LineSegment{

    LineSegment(cv::Point2d start,cv::Point2d end);
    LineSegment(double x1, double y1, double x2, double y2);
    LineSegment(cv::Vec4d vec={});

    cv::Vec3d getLineHomogeneousCoordinates() const;
    cv::Point2d getVector() const;


    //lets pretend that start!=end is always true
    cv::Point2d start;
    cv::Point2d end;

    unsigned int vp_id;
};

struct PlaneInfo{
    PlaneInfo(){}
    std::vector<LineSegment> inliners;
    std::vector<std::vector<int>> adjacency_graph;
    double a;
    double b;
};



cv::Point2d getIntersection(const LineSegment& lhs, const LineSegment& rhs);
bool areAdjacent(const LineSegment& lhs, const LineSegment& rhs);

std::vector<std::vector<int>> findAdjacencyGraph(std::vector<LineSegment>& lines);

std::vector<cv::Point3d> findVanishingPoints(std::vector<LineSegment>& lines);
std::vector<std::vector<uint32_t>> findOrthMat(const std::vector<std::vector<int>>& adjacency_graph, const std::vector<LineSegment>& lines, const std::vector<cv::Point3d>& vanishing_points);

double estimateFocalLength(const std::vector<cv::Point3d>& vanishing_points,const std::vector<std::vector<uint32_t>>& orth_mat);

casadi::MX xRotationMat( const casadi::MX& a);
casadi::MX yRotationMat(const casadi::MX& b);
casadi::MX transformationMatInvT(const casadi::MX& a,const casadi::MX& b,const casadi::MX& f);
casadi::MX transformationMatInvT(const casadi::MX& a,const casadi::MX& b,const casadi::MX& f);
casadi::MX getOrthoganalityMetric(const LineSegment& li,const LineSegment& lj, double f_val);

std::pair<double,double> find_ab(const LineSegment& li,const LineSegment& lj, double f_val);


std::pair<double,double> findDominantPlane(std::vector<uint8_t>& inliners, const std::vector<LineSegment>& lines, const std::vector<std::vector<int>>& adjacency_graph, double f, double t = 1e-2, int N = 500);

void remove_lines(std::vector<LineSegment>& lines, std::vector<std::vector<int>>& adjacency_graph, const std::vector<uint8_t>& mask, bool remove_zeros);

std::vector<PlaneInfo> getPlanes(const std::vector<LineSegment>& lines, const std::vector<std::vector<int>>& adjacency_graph, double f, double t = 1e-2, int N = 500);