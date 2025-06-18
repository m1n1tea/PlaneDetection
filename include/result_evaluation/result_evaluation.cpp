#include <result_evaluation/result_evaluation.h>

#include <map>
#include <set>
#include <sstream>

namespace
{
typedef cv::Point3_<uint8_t> Pixel;

std::string getHexString(Pixel color)
{
    int blue = color.x;
    int green = color.y;
    int red = color.z;

    std::ostringstream s_stream;
    s_stream << std::hex << "#";
    if(red<16)
        s_stream << "0";
    s_stream << red;
    if(green<16)
        s_stream << "0";
    s_stream << green;
    if(blue<16)
        s_stream << "0";
    s_stream << blue;

    return s_stream.str();
}

}  // namespace

EvaluationResult evaluateResult(cv::Mat predicted_planes, cv::Mat ground_truth_planes)
{
    assert(predicted_planes.type() == CV_8UC3);
    assert(ground_truth_planes.type() == CV_8UC3);
    assert(predicted_planes.size() == ground_truth_planes.size());

    std::map<std::string, std::map<std::string, int>> planes_intersection;
    std::map<std::string, std::map<std::string, int>> planes_union;
    std::set<std::pair<std::string, std::string>> intersecting_planes;
    EvaluationResult result;

    Pixel background(0, 0, 0);

    int n = predicted_planes.rows;
    int m = predicted_planes.cols;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            Pixel ground_truth_pixel = ground_truth_planes.at<Pixel>(i, j);
            Pixel predicted_pixel = predicted_planes.at<Pixel>(i, j);
            if (ground_truth_pixel == background && predicted_pixel == background)
            {
                continue;
            }
            if (ground_truth_pixel == background && predicted_pixel != background)
            {
                result.false_positive_pixels++;
                continue;
            }
            if (ground_truth_pixel != background && predicted_pixel == background)
            {
                result.false_negative_pixels++;
                continue;
            }
            std::string ground_truth_str = getHexString(ground_truth_pixel);
            std::string predicted_str = getHexString(predicted_pixel);
            intersecting_planes.emplace(ground_truth_str, predicted_str);
            planes_intersection[ground_truth_str][predicted_str] = 0;
            planes_union[ground_truth_str][predicted_str] = 0;
            intersecting_planes.emplace(ground_truth_str, predicted_str);
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            Pixel ground_truth_pixel = ground_truth_planes.at<Pixel>(i, j);
            Pixel predicted_pixel = predicted_planes.at<Pixel>(i, j);
            if (ground_truth_pixel == background || predicted_pixel == background)
            {
                continue;
            }
            std::string ground_truth_str = getHexString(ground_truth_pixel);
            std::string predicted_str = getHexString(predicted_pixel);
            planes_intersection[ground_truth_str][predicted_str] += 1;

            for (auto& [predicted_plane_color, count] : planes_union[ground_truth_str])
            {
                count++;
            }

            for (auto& [ground_truth_color, predicted_planes] : planes_union)
            {
                if (auto it = predicted_planes.find(predicted_str); it != predicted_planes.end())
                {
                    it->second++;
                }
            }
            planes_union[ground_truth_str][predicted_str]--;
        }
    }

    std::map<std::string, std::pair<double, std::string>> ground_truths_best_predicted_plane;
    for (auto& [ground_truth_color, predicted_color] : intersecting_planes)
    {
        double plane_intersection = planes_intersection[ground_truth_color][predicted_color];
        double plane_union = planes_union[ground_truth_color][predicted_color];
        ground_truths_best_predicted_plane[ground_truth_color] =
            std::max(ground_truths_best_predicted_plane[ground_truth_color],
                     std::make_pair(plane_intersection / plane_union, predicted_color));
    }

    std::map<std::string, std::pair<double, std::string>> predicted_planes_best_ground_truth;
    for (auto& [ground_truth_color, predicted_color] : intersecting_planes)
    {
        double iou=-1;
        if (ground_truths_best_predicted_plane[ground_truth_color].second == predicted_color)
        {
            iou=ground_truths_best_predicted_plane[ground_truth_color].first;
        }
        predicted_planes_best_ground_truth[predicted_color] =
            std::max(predicted_planes_best_ground_truth[predicted_color],
                     std::make_pair(iou, ground_truth_color));
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            Pixel ground_truth_pixel = ground_truth_planes.at<Pixel>(i, j);
            Pixel predicted_pixel = predicted_planes.at<Pixel>(i, j);
            if (ground_truth_pixel == background || predicted_pixel == background)
            {
                continue;
            }
            std::string ground_truth_str = getHexString(ground_truth_pixel);
            std::string predicted_str = getHexString(predicted_pixel);
            if (predicted_planes_best_ground_truth[predicted_str].second == ground_truth_str)
            {
                result.correct_predicted_pixels++;
            }
            else
            {
                result.incorrect_predicted_pixels++;
            }
        }
    }



    for (const auto& el : predicted_planes_best_ground_truth)
    {
        result.iou_per_ground_truth_plane.emplace_back(el.second.second, el.first, el.second.first);
    }

    return result;
}


nlohmann::json EvaluationResult::toJson(){
    nlohmann::json result;

    result["correct_predicted_pixels"] = correct_predicted_pixels;
    result["incorrect_predicted_pixels"] = incorrect_predicted_pixels;
    result["false_negative_pixels"] = false_negative_pixels;
    result["false_positive_pixels"] = false_positive_pixels;
    result["ground_truth_labels"] = nlohmann::json::array();
    result["f1_plane_deteciton"] = (2.0 * (correct_predicted_pixels + incorrect_predicted_pixels)) / (2 * (correct_predicted_pixels + incorrect_predicted_pixels) + false_negative_pixels + false_positive_pixels);
    result["f1_plane_separation"] = (2.0 * (correct_predicted_pixels)) / (2 * (correct_predicted_pixels) + false_negative_pixels + false_positive_pixels + incorrect_predicted_pixels);


    for(const auto& [ground_truth_color, predicted_color, iou] : iou_per_ground_truth_plane){
        nlohmann::json plane;
        plane["plane_color"] = ground_truth_color;
        plane["corresponding_predicted_plane_color"] = predicted_color;
        plane["intersection_over_union"] = iou;
        result["ground_truth_labels"].push_back(plane);
    }
    return result;
}