#include <VPCluster.h>
#include <VPSample.h>
#include <focal_length_estimation/focal_length_estimation.h>

std::vector<cv::Point3d> findVanishingPoints(std::vector<LineSegment> &lines)
{
    std::vector<std::vector<float> *> pts;

    for (size_t i = 0; i < lines.size(); ++i)
    {
        std::vector<float> *p = new std::vector<float>(4);
        (*p)[0] = lines[i].start.x;
        (*p)[1] = lines[i].start.y;
        (*p)[2] = lines[i].end.x;
        (*p)[3] = lines[i].end.y;
        pts.push_back(p);
    }

    std::vector<unsigned int> Labels;
    std::vector<unsigned int> LabelCount;
    std::vector<unsigned int> modelIndex;
    std::vector<std::vector<float> *> *mModels = VPSample::run(&pts, 5000, 2, 0, 1);
    int classNum = VPCluster::run(Labels, LabelCount, modelIndex, &pts, mModels, 1.0, 2);

    std::vector<cv::Point3d> vps(modelIndex.size());

    for (size_t i = 0; i < modelIndex.size(); ++i)
    {
        const std::vector<float> &vp = *((*mModels)[modelIndex[i]]);
        vps[i] = cv::Point3d(vp[0], vp[1], vp[2]);
    }
    for (size_t i = 0; i < Labels.size(); ++i)
    {
        lines[i].vp_id = Labels[i];
    }
    for (size_t i = 0; i < mModels->size(); ++i) delete (*mModels)[i];
    delete mModels;
    for (size_t i = 0; i < pts.size(); i++) delete pts[i];

    return vps;
}

bool possibleOrth(cv::Point3d vpi, cv::Point3d vpj)
{
    vpi /= norm(vpi);
    vpj /= norm(vpj);
    return vpi.z * vpj.z * (vpi.x * vpj.x + vpi.y * vpj.y) < 0;
}

std::vector<std::vector<uint32_t>> findOrthMat(const std::vector<std::vector<int>> &adjacency_graph,
                                               const std::vector<LineSegment> &lines,
                                               const std::vector<cv::Point3d> &vanishing_points,
                                               double vanishing_points_relative_threshold,
                                               double vanishing_points_absolute_threshold)
{
    std::vector<std::vector<uint32_t>> mat(vanishing_points.size(), std::vector<uint32_t>(vanishing_points.size()));
    int sum = 0;
    uint32_t max_val = 1;

    for (size_t i = 0; i < adjacency_graph.size(); ++i)
    {
        for (int j : adjacency_graph[i])
        {
            unsigned int vp_id1 = lines[i].vp_id;
            unsigned int vp_id2 = lines[j].vp_id;
            if (!possibleOrth(vanishing_points[vp_id1], vanishing_points[vp_id2]))
            {
                continue;
            }
            sum += 1;
            mat[vp_id1][vp_id2] += 1;
            mat[vp_id2][vp_id1] += 1;
            max_val = std::max(mat[vp_id1][vp_id2], max_val);
        }
    }
    for (int i = 0; i < vanishing_points.size(); ++i)
    {
        for (int j = 0; j < vanishing_points.size(); ++j)
        {
            if (mat[i][j] == max_val || (mat[i][j] >= vanishing_points_absolute_threshold &&
                                         ((double)mat[i][j]) / sum >= vanishing_points_relative_threshold))
            {
                mat[i][j] = 1;
            }
            else
            {
                mat[i][j] = 0;
            }
        }
    }

    return mat;
}

double estimateFocalLength(const std::vector<cv::Point3d> &vanishing_points,
                           const std::vector<std::vector<uint32_t>> &orth_mat)
{
    double num_sum = 0;
    double denom_sum = 0;

    for (size_t i = 0; i < vanishing_points.size(); ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            if (orth_mat[i][j] == 0)
            {
                continue;
            }
            cv::Point3d vpi = vanishing_points[i];
            cv::Point3d vpj = vanishing_points[j];
            vpi /= norm(vpi);
            vpj /= norm(vpj);

            num_sum += vpi.z * vpj.z * (vpi.x * vpj.x + vpi.y * vpj.y);
            denom_sum += vpi.z * vpj.z * vpi.z * vpj.z;
        }
    }
    if (denom_sum == 0)
    {
        return 0;
    }
    return std::sqrt(-num_sum / denom_sum);
}

double findFocalLength(const std::vector<LineSegment> &lines, double relative_dist_threshold, double angle_threshold,
                       double vanishing_points_relative_threshold, double vanishing_points_absolute_threshold)
{
    std::vector<LineSegment> lines_copy = lines;
    std::vector<std::vector<int>> adjacency_graph =
        findAdjacencyGraph(lines_copy, relative_dist_threshold, angle_threshold);

    std::vector<uint8_t> singleLines(lines_copy.size(), 0);
    for (int i = 0; i < lines_copy.size(); ++i)
    {
        singleLines[i] = adjacency_graph[i].empty();
    }
    remove_lines(lines_copy, adjacency_graph, singleLines, false);

    std::vector<cv::Point3d> vps = findVanishingPoints(lines_copy);
    std::vector<std::vector<uint32_t>> orth_mat = findOrthMat(
        adjacency_graph, lines_copy, vps, vanishing_points_relative_threshold, vanishing_points_absolute_threshold);
    return estimateFocalLength(vps, orth_mat);
}