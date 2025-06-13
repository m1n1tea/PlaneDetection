#include <plane_labeling/plane_labeling.h>

#include <algorithm>
#include <opencv2/imgproc.hpp>

void distributeLinesBeetweenPlanes(std::vector<PlaneInfo> &planes, const std::vector<LineSegment> &lines, double f,
                                   double t, double relative_dist_threshold, double angle_threshold)
{
    std::vector<std::vector<int>> adjacency_graph = findAdjacencyGraph(lines, relative_dist_threshold, angle_threshold);
    int n = lines.size();
    std::vector<std::vector<std::tuple<int, int, double>>> best_line_pair_planes(n);
    for (int i = 0; i < n; ++i)
    {
        for (int j : adjacency_graph[i])
        {
            best_line_pair_planes[i].emplace_back(j, -1, t);
        }
    }

    for (int p = 0; p < planes.size(); ++p)
    {
        auto &plane = planes[p];
        plane.adjacency_graph.assign(n, {});
        std::vector<double> ab{plane.a, plane.b};
        for (int i = 0; i < n; ++i)
        {
            for (auto &line_pair : best_line_pair_planes[i])
            {
                int j = std::get<0>(line_pair);
                int &best_p = std::get<1>(line_pair);
                double &best_metric = std::get<2>(line_pair);
                double metric = OrthoganalityMetricFunction(lines[i], lines[j], f).calc(ab.data());
                if (metric < best_metric)
                {
                    best_metric = metric;
                    best_p = p;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (auto &line_pair : best_line_pair_planes[i])
        {
            int j = std::get<0>(line_pair);
            int p = std::get<1>(line_pair);
            if (p != -1)
            {
                planes[p].adjacency_graph[i].push_back(j);
            }
        }
    }
}

LineSegment getLongestSegment(cv::Point2d a, cv::Point2d b, cv::Point2d c)
{
    LineSegment ans(a, b);
    double dist = cv::norm(a - b);
    if (cv::norm(a - c) > dist)
    {
        dist = cv::norm(a - c);
        ans = LineSegment(a, c);
    }
    if (cv::norm(b - c) > dist)
    {
        dist = cv::norm(b - c);
        ans = LineSegment(b, c);
    }
    return ans;
}

cv::Vec3d getParallelLine(cv::Vec3d line, cv::Point2d pt)
{
    return {line[0], line[1], -(line[0] * pt.x + line[1] * pt.y)};
}

std::vector<cv::Point2f> getRect(const LineSegment &li, const LineSegment &lj, double f, double a, double b)
{
    cv::Matx33d H = OrthoganalityMetricFunction::transformationMat(a, b, f);
    cv::Matx33d H_inv = OrthoganalityMetricFunction::transformationMatInv(a, b, f);

    cv::Point2d intersection = getIntersection(li, lj);

    LineSegment li_true = getLongestSegment(li.start, li.end, intersection);
    LineSegment lj_true = getLongestSegment(lj.start, lj.end, intersection);

    std::vector<cv::Vec3d> rect_points_homo(4);
    std::vector<cv::Point2d> rect_points(4);

    rect_points[0] = li_true.start;
    rect_points[1] = li_true.end;
    rect_points[2] = lj_true.start;
    rect_points[3] = lj_true.end;

    for (int i = 0; i < 4; ++i)
    {
        rect_points_homo[i] = cv::Vec3d(rect_points[i].x, rect_points[i].y, 1);
        rect_points_homo[i] = H * rect_points_homo[i];
        rect_points_homo[i] /= rect_points_homo[i][2];
        rect_points[i] = cv::Point2d(rect_points_homo[i][0], rect_points_homo[i][1]);
    }

    cv::Vec3d line_i = H_inv.t() * li.getLineHomogeneousCoordinates();
    cv::Vec3d line_j = H_inv.t() * lj.getLineHomogeneousCoordinates();

    cv::Vec3d line1 = getParallelLine(line_i, rect_points[2]);
    cv::Vec3d line2 = getParallelLine(line_i, rect_points[3]);
    cv::Vec3d line3 = getParallelLine(line_j, rect_points[0]);
    cv::Vec3d line4 = getParallelLine(line_j, rect_points[1]);

    std::vector<cv::Point2f> rect_corners(4);
    std::vector<cv::Vec3d> rect_corners_homo(4);

    rect_corners_homo[0] = line1.cross(line3);
    rect_corners_homo[1] = line1.cross(line4);
    rect_corners_homo[2] = line2.cross(line4);
    rect_corners_homo[3] = line2.cross(line3);

    for (int i = 0; i < 4; ++i)
    {
        rect_corners_homo[i] = H_inv * rect_corners_homo[i];
        rect_corners_homo[i] /= rect_corners_homo[i][2];
        rect_corners[i] = cv::Point2f(rect_corners_homo[i][0], rect_corners_homo[i][1]);
    }

    return rect_corners;
}

bool cmpPoints(const std::pair<cv::Point2f, int> &lhs, const std::pair<cv::Point2f, int> &rhs)
{
    return std::tie(lhs.first.x, lhs.first.y) < std::tie(rhs.first.x, rhs.first.y);
}

float findGoodnessScore(const std::vector<cv::Point2f> &polygon, int plane,
                        const std::vector<std::pair<cv::Point2f, int>> &intersection_points)
{
    int num = 0;
    int denom = 0;

    std::pair<cv::Point2f, int> left_bound_val(polygon[0], plane);
    std::pair<cv::Point2f, int> right_bound_val(polygon[0], plane);
    for (const cv::Point2f &pt : polygon)
    {
        left_bound_val.first.x = std::min(left_bound_val.first.x, pt.x);
        left_bound_val.first.y = std::min(left_bound_val.first.y, pt.y);
        right_bound_val.first.x = std::max(right_bound_val.first.x, pt.x);
        right_bound_val.first.y = std::max(right_bound_val.first.y, pt.y);
    }
    auto left_bound =
        std::lower_bound(intersection_points.begin(), intersection_points.end(), left_bound_val, cmpPoints);
    auto right_bound =
        std::upper_bound(intersection_points.begin(), intersection_points.end(), right_bound_val, cmpPoints);

    for (auto it = left_bound; it != right_bound; it++)
    {
        cv::Point2f pt = it->first;
        int point_plane = it->second;
        if (cv::pointPolygonTest(polygon, pt, false) >= 0)
        {
            denom += 1;
            if (point_plane == plane)
            {
                num += 1;
            }
        }
    }
    if (denom == 0)
    {
        return 0;
    }
    else
    {
        return num / denom;
    }
}

bool areConvexPolygonsIntersect(const std::vector<cv::Point2f> &poly1, const std::vector<cv::Point2f> &poly2)
{
    std::vector<cv::Point2f> intersection;
    float area = cv::intersectConvexConvex(poly1, poly2, intersection);
    return area > 0;
}

float getMinX(const std::vector<cv::Point2f> &polygon)
{
    float min_x(polygon[0].x);
    for (const cv::Point2f &pt : polygon)
    {
        min_x = std::min(min_x, pt.x);
    }
    return min_x;
}

float getMaxX(const std::vector<cv::Point2f> &polygon)
{
    float max_x(polygon[0].x);
    for (const cv::Point2f &pt : polygon)
    {
        max_x = std::max(max_x, pt.x);
    }
    return max_x;
}

std::vector<std::set<int>> findConflictGraph(const std::vector<std::vector<cv::Point2f>> &polygons,
                                             const std::vector<int> &polygons_plane, int planes_count)
{
    int n = polygons.size();
    std::vector<std::set<int>> graph(n);

    std::vector<std::tuple<float, float, int>> sorted_left_bounds(n);

    for (int i = 0; i < n; ++i)
    {
        float min_x = getMinX(polygons[i]);
        float max_x = getMaxX(polygons[i]);
        sorted_left_bounds[i] = std::tie(min_x, max_x, i);
    }
    std::sort(sorted_left_bounds.begin(), sorted_left_bounds.end());

    int i = 0;
    std::set<std::pair<float, int>> right_bounds;

    for (const auto &el : sorted_left_bounds)
    {
        float l = std::get<0>(el);
        float r = std::get<1>(el);
        int i = std::get<2>(el);

        while (!right_bounds.empty() && l > right_bounds.begin()->first)
        {
            right_bounds.erase(right_bounds.begin());
        }
        for (const auto &pr : right_bounds)
        {
            int j = pr.second;
            if (polygons_plane[i] != polygons_plane[j] && areConvexPolygonsIntersect(polygons[i], polygons[j]))
            {
                graph[i].insert(j);
                graph[j].insert(i);
            }
        }

        right_bounds.emplace(r, i);
    }

    return graph;
}

void drawContour4(cv::Mat &image, std::vector<cv::Point2f> vertices2f, cv::Scalar color)
{
    std::vector<std::vector<cv::Point>> vertices(1, std::vector<cv::Point>(4));
    for (int i = 0; i < 4; ++i)
    {
        vertices[0][i] = vertices2f[i];
    }
    cv::drawContours(image, vertices, 0, color, -1);
}

std::vector<cv::Mat> findPlaneRegions(const std::vector<PlaneInfo> &planes, const std::vector<LineSegment> &lines,
                                      double f, cv::Size img_size)
{
    if (planes.empty())
    {
        return {};
    }
    std::vector<std::vector<cv::Point2f>> polygons;
    std::vector<std::pair<cv::Point2f, int>> intersection_points;
    std::vector<int> polygons_plane;
    std::vector<float> polygons_goodness;

    int polygons_count = 0;
    for (int p = 0; p < planes.size(); ++p)
    {
        for (int i = 0; i < lines.size(); ++i)
        {
            polygons_count += planes[p].adjacency_graph[i].size();
        }
    }
    polygons_count /= 2;  // every line pair is added 2 times
    polygons.reserve(polygons_count);
    intersection_points.reserve(polygons_count);
    polygons_plane.reserve(polygons_count);
    polygons_goodness.reserve(polygons_count);

    for (int p = 0; p < planes.size(); ++p)
    {
        for (int i = 0; i < lines.size(); ++i)
        {
            for (int j : planes[p].adjacency_graph[i])
            {
                if (i >= j)
                {
                    continue;
                }
                intersection_points.emplace_back(cv::Point2f(getIntersection(lines[i], lines[j])), p);
                polygons.push_back(getRect(lines[i], lines[j], f, planes[p].a, planes[p].b));
                polygons_plane.push_back(p);
            }
        }
    }

    std::sort(intersection_points.begin(), intersection_points.end(), cmpPoints);
    for (int i = 0; i < polygons_count; ++i)
    {
        polygons_goodness.push_back(findGoodnessScore(polygons[i], polygons_plane[i], intersection_points));
    }

    std::vector<std::set<int>> graph = findConflictGraph(polygons, polygons_plane, planes.size());
    std::set<std::tuple<float, int, int>> polygon_scores_set;
    std::vector<std::pair<float, int>> polygon_scores_vec(polygons_count);

    for (int i = 0; i < polygons_count; ++i)
    {
        float score = 0;
        int confilcts = 0;
        for (int j : graph[i])
        {
            score += polygons_goodness[j];
            confilcts += 1;
        }
        polygon_scores_set.insert(std::tie(score, confilcts, i));
        polygon_scores_vec[i] = std::make_pair(score, confilcts);
    }

    auto current_worst_rect = polygon_scores_set.rbegin();

    while (std::get<1>(*current_worst_rect) > 0)
    {
        int i = std::get<2>(*current_worst_rect);
        polygon_scores_set.erase(*current_worst_rect);
        for (int j : graph[i])
        {
            // remove rect from overlapping rects
            float &score = polygon_scores_vec[j].first;
            int &conflicts = polygon_scores_vec[j].second;
            polygon_scores_set.erase(std::tie(score, conflicts, j));
            score -= polygons_goodness[i];
            conflicts -= 1;
            polygon_scores_set.insert(std::tie(score, conflicts, j));
            graph[j].erase(i);
        }
        graph[i].clear();
        current_worst_rect = polygon_scores_set.rbegin();
    }

    std::vector<cv::Mat> planes_mask(planes.size());

    for (int p = 0; p < planes_mask.size(); ++p)
    {
        planes_mask[p] = cv::Mat(img_size, CV_8U, cv::Scalar(0));
    }
    for (auto el : polygon_scores_set)
    {
        int i = std::get<2>(el);
        drawContour4(planes_mask[polygons_plane[i]], polygons[i], cv::Scalar(255));
    }
    return planes_mask;
}
