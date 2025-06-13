#include <plane_orientation_detection/plane_orientation_detection.h>

#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <set>

namespace
{
std::random_device rd;
}

std::vector<double> find_ab(const LineSegment &segment_i, const LineSegment &segment_j, double f_val)
{
    std::vector<double> ab(2, 0);

    cv::Ptr<OrthoganalityMetricFunction> func = cv::makePtr<OrthoganalityMetricFunction>(segment_i, segment_j, f_val);

    double step = 0.2;

    int max_size = 30;

    std::set<std::array<double, 3>> potentional_abs;

    for (double a = 0; a < 2 * std::numbers::pi; a += step)
    {
        for (double b = 0; b < 2 * std::numbers::pi; b += step)
        {
            std::vector<double> local_ab{a, b};
            double val = func->calc(local_ab.data());
            potentional_abs.insert({val, a, b});
            if (potentional_abs.size() > max_size)
            {
                potentional_abs.erase(--potentional_abs.end());
            }
        }
    }

    cv::Mat solver_step = (cv::Mat_<double>(2, 1) << 0.05, 0.05);

    cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create(func, solver_step);

    int ind = 0;

    double best_val = 2;
    for (const auto &potential_vec : potentional_abs)
    {
        std::vector<double> potential_ab = {potential_vec[1], potential_vec[2]};
        double val = solver->minimize(potential_ab);

        if (val < best_val)
        {
            best_val = val;
            ab = potential_ab;
        }
        ind++;
    }

    return ab;
}

int count_nonzeros(const std::vector<uint8_t> &mask)
{
    int sum = 0;
    for (uint8_t el : mask)
    {
        if (el != 0)
        {
            sum += 1;
        }
    }
    return sum;
}

std::vector<double> findDominantPlane(std::vector<uint8_t> &inliners, const std::vector<LineSegment> &lines,
                                      const std::vector<std::vector<int>> &adjacency_graph, double f, double t, int N)
{
    int n = lines.size();
    int edges_count = 0;

    for (int i = 0; i < n; ++i)
    {
        edges_count += adjacency_graph[i].size();
    }
    if (edges_count == 0)
    {
        return {};
    }

    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, edges_count - 1);

    std::vector<double> best_ab;

    for (int o = 0; o < N; ++o)
    {
        int edge_num = dist(gen);
        int v1 = -1;
        int v2 = -1;
        for (int i = 0; i < n; ++i)
        {
            if (edge_num >= adjacency_graph[i].size())
            {
                edge_num -= adjacency_graph[i].size();
            }
            else
            {
                v1 = i;
                v2 = edge_num;
                break;
            }
        }
        std::vector<double> ab = find_ab(lines[v1], lines[v2], f);
        std::vector<uint8_t> local_inliners(n, 0);
        for (int i = 0; i < n; ++i)
        {
            for (int j : adjacency_graph[i])
            {
                if (local_inliners[i] != 0 && local_inliners[j] != 0)
                {
                    continue;
                }
                double metric = OrthoganalityMetricFunction(lines[i], lines[j], f).calc(ab.data());

                if (metric < t)
                {
                    local_inliners[i] = 1;
                    local_inliners[j] = 1;
                }
            }
        }
        if (count_nonzeros(local_inliners) > count_nonzeros(inliners))
        {
            inliners = std::move(local_inliners);
            best_ab = ab;
        }
    }
    return best_ab;
}

std::vector<PlaneInfo> getPlanes(const std::vector<LineSegment> &lines, double f, double t, int N,
                                 double relative_dist_threshold, double angle_threshold,
                                 double plane_relative_threshold, double plane_absolute_threshold)
{
    auto lines_copy = lines;
    std::vector<std::vector<int>> adjacency_graph =
        findAdjacencyGraph(lines_copy, relative_dist_threshold, angle_threshold);

    std::vector<uint8_t> singleLines(lines_copy.size(), 0);
    for (int i = 0; i < lines_copy.size(); ++i)
    {
        singleLines[i] = adjacency_graph[i].empty();
    }

    remove_lines(lines_copy, adjacency_graph, singleLines, false);
    double starting_lines_count = lines_copy.size();

    std::vector<PlaneInfo> planes;

    do
    {
        int n = lines_copy.size();
        int pairs_count = 0;
        for (int i = 0; i < n; ++i)
        {
            pairs_count += adjacency_graph[i].size();
        }
        PlaneInfo plane;
        std::vector<uint8_t> inliners;
        std::vector<double> ab = findDominantPlane(inliners, lines_copy, adjacency_graph, f, t, N);
        if (ab.empty())
        {
            break;
        }
        plane.a = ab[0];
        plane.b = ab[1];
        remove_lines(lines_copy, adjacency_graph, inliners, false);
        int inliners_count = count_nonzeros(inliners);

        if (inliners_count >= plane_absolute_threshold &&
            ((double)inliners_count) / starting_lines_count >= plane_relative_threshold)
        {
            planes.push_back(plane);
        }
        else
        {
            break;
        }

    } while (true);

    return planes;
}