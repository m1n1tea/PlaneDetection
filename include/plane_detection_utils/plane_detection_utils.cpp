#include <plane_detection_utils/plane_detection_utils.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <opencv2/imgproc.hpp>

namespace
{
cv::Point2d inf = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
}

LineSegment::LineSegment(cv::Point2d start, cv::Point2d end) : start(start), end(end), vp_id(-1) {}
LineSegment::LineSegment(double x1, double y1, double x2, double y2) : start(x1, y1), end(x2, y2), vp_id(-1) {}
LineSegment::LineSegment(cv::Vec4f vec) : start(vec[0], vec[1]), end(vec[2], vec[3]), vp_id(-1) {}

cv::Vec3d LineSegment::getLineHomogeneousCoordinates() const
{
    cv::Vec3d p1 = {start.x, start.y, 1};
    cv::Vec3d p2 = {end.x, end.y, 1};
    return p1.cross(p2);
}

cv::Point2d LineSegment::getVector() const { return end - start; }

cv::Point2d getIntersection(const LineSegment &lhs, const LineSegment &rhs)
{
    cv::Vec3d hom_point = lhs.getLineHomogeneousCoordinates().cross(rhs.getLineHomogeneousCoordinates());
    if (hom_point[2] == 0)
    {
        return inf;
    }
    hom_point /= hom_point[2];
    return {hom_point[0], hom_point[1]};
}

bool areAdjacent(const LineSegment &lhs, const LineSegment &rhs, double relative_dist_threshold, double angle_threshold)
{
    double cos_threshold = std::cos(angle_threshold);

    cv::Point2d vec1 = lhs.getVector();
    cv::Vec2d vec2 = rhs.getVector();
    double line_len1 = norm(vec1);
    double line_len2 = norm(vec2);
    vec1 /= line_len1;
    vec2 /= line_len2;
    double cos = std::abs(vec1.ddot(vec2));
    if (cos > cos_threshold)
    {
        return false;
    }
    cv::Point2d int_point = getIntersection(lhs, rhs);
    if (int_point == inf)
    {
        return false;
    }
    double line_with_intersection_len1 = std::max(norm(lhs.end - int_point), norm(lhs.start - int_point));
    double line_with_intersection_len2 = std::max(norm(rhs.end - int_point), norm(rhs.start - int_point));

    if (line_with_intersection_len1 / line_len1 > relative_dist_threshold ||
        line_with_intersection_len2 / line_len2 > relative_dist_threshold)
    {
        return false;
    }
    return true;
}

std::vector<LineSegment> detectLines(cv::Mat src)
{
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector();

    cv::Mat grayImage;

    cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);

    std::vector<cv::Vec4f> detected_lines;
    lsd->detect(grayImage, detected_lines);
    int n = detected_lines.size();
    std::vector<LineSegment> lines;
    lines.reserve(n);
    for (size_t i = 0; i < n; i++)
    {
        lines.emplace_back(detected_lines[i]);
    }
    return lines;
}

std::vector<std::vector<int>> findAdjacencyGraph(const std::vector<LineSegment> &lines, double relative_dist_threshold,
                                                 double angle_threshold)
{
    int n = lines.size();
    std::vector<std::vector<int>> graph(n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (areAdjacent(lines[i], lines[j], relative_dist_threshold, angle_threshold))
            {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }
    return graph;
}

void remove_lines(std::vector<LineSegment> &lines, std::vector<std::vector<int>> &adjacency_graph,
                  const std::vector<uint8_t> &mask, bool remove_zeros)
{
    int n = lines.size();
    std::vector<int> transformation(n, -1);
    int new_i = 0;
    for (int i = 0; i < n; ++i)
    {
        if ((mask[i] == 0) == remove_zeros)
        {
            // remove this element
            continue;
        }
        transformation[i] = new_i;
        lines[new_i] = std::move(lines[i]);
        adjacency_graph[new_i] = std::move(adjacency_graph[i]);
        new_i++;
    }
    n = new_i;
    lines.resize(n);
    adjacency_graph.resize(n);
    for (int i = 0; i < n; ++i)
    {
        for (int &j : adjacency_graph[i])
        {
            j = transformation[j];
        }
        auto ptr_begin = std::remove(adjacency_graph[i].begin(), adjacency_graph[i].end(), -1);
        auto ptr_end = adjacency_graph[i].end();
        adjacency_graph[i].erase(ptr_begin, ptr_end);
    }
}

OrthoganalityMetricFunction::OrthoganalityMetricFunction(const LineSegment &li, const LineSegment &lj, double f_val)
    : li(li.getLineHomogeneousCoordinates()), lj(lj.getLineHomogeneousCoordinates()), f_val(f_val)
{
}

int OrthoganalityMetricFunction::getDims() const { return 2; }  // 2D problem

cv::Matx33d OrthoganalityMetricFunction::xRotationMat(double a)
{
    double cos_a = std::cos(a);
    double sin_a = std::sin(a);
    cv::Matx33d rotation_mat = cv::Matx33d::zeros();

    rotation_mat(0, 0) = 1;
    rotation_mat(1, 1) = cos_a;
    rotation_mat(1, 2) = -sin_a;
    rotation_mat(2, 1) = sin_a;
    rotation_mat(2, 2) = cos_a;

    return rotation_mat;
}

cv::Matx33d OrthoganalityMetricFunction::yRotationMat(double b)
{
    double cos_b = std::cos(b);
    double sin_b = std::sin(b);
    cv::Matx33d rotation_mat = cv::Matx33d::zeros();

    rotation_mat(1, 1) = 1;
    rotation_mat(0, 0) = cos_b;
    rotation_mat(0, 2) = -sin_b;
    rotation_mat(2, 0) = sin_b;
    rotation_mat(2, 2) = cos_b;

    return rotation_mat;
}

cv::Matx33d OrthoganalityMetricFunction::calibrationMat(double f)
{
    cv::Matx33d calibration_mat = cv::Matx33d::zeros();
    calibration_mat(0, 0) = f;
    calibration_mat(1, 1) = f;
    calibration_mat(2, 2) = 1;
    return calibration_mat;
}

cv::Matx33d OrthoganalityMetricFunction::calibrationMatInv(double f)
{
    cv::Matx33d calibration_mat_inv = cv::Matx33d::zeros();
    calibration_mat_inv(0, 0) = 1 / f;
    calibration_mat_inv(1, 1) = 1 / f;
    calibration_mat_inv(2, 2) = 1;
    return calibration_mat_inv;
}

cv::Matx33d OrthoganalityMetricFunction::transformationMat(double a, double b, double f)
{
    return xRotationMat(a) * yRotationMat(b) * calibrationMatInv(f);
}

cv::Matx33d OrthoganalityMetricFunction::transformationMatInv(double a, double b, double f)
{
    return calibrationMat(f) * yRotationMat(b).t() * xRotationMat(a).t();
}

cv::Matx33d OrthoganalityMetricFunction::transformationMatInvT(double a, double b, double f)
{
    return xRotationMat(a) * yRotationMat(b) * calibrationMat(f);
}

double OrthoganalityMetricFunction::calc(const double *x) const
{
    cv::Matx33d H = transformationMatInvT(x[0], x[1], f_val);
    cv::Vec3d Hli = H * li;
    cv::Vec3d Hlj = H * lj;

    cv::Vec2d vi{Hli[0], Hli[1]};
    cv::Vec2d vj{Hlj[0], Hlj[1]};

    double eps = 1e-10;

    double ans = vi.dot(vj) * vi.dot(vj) / (vi.dot(vi) * vj.dot(vj) + eps);

    return ans;
}