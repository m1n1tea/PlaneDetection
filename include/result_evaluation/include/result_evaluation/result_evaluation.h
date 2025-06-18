

#ifndef PLANE_RESULT_EVALUATION
#define PLANE_RESULT_EVALUATION

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <utility>
#include <vector>



struct EvaluationResult
{
    int correct_predicted_pixels = 0;
    int incorrect_predicted_pixels = 0;
    int false_negative_pixels = 0;
    int false_positive_pixels = 0;
    std::vector<std::tuple<std::string, std::string, double>> iou_per_ground_truth_plane;

    nlohmann::json toJson();
};

EvaluationResult evaluateResult(cv::Mat predicted_planes, cv::Mat ground_truth_planes);

#endif  // PLANE_RESULT_EVALUATION