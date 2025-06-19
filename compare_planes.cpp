#include <result_evaluation/result_evaluation.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    const std::string keys =
        "{help h usage ?       |      | print this message }"
        "{@predicted_image     |<none>| predicted image path }"
        "{@ground_truth_image  |<none>| ground truth image path }";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help") || argc < 3)
    {
        parser.printMessage();
        return 0;
    }

    std::string predicted_image_path = parser.get<cv::String>("@predicted_image");
    if (!fs::is_regular_file(predicted_image_path))
    {
        std::cout << "invalid predictede image path\n";
        return 1;
    }

    std::string ground_trurh_image_path = parser.get<cv::String>("@ground_truth_image");
    if (!fs::is_regular_file(ground_trurh_image_path))
    {
        std::cout << "invalid ground truth image path\n";
        return 1;
    }

    cv::Mat predicted_img = cv::imread(predicted_image_path, cv::IMREAD_COLOR);
    cv::Mat ground_truth_img = cv::imread(ground_trurh_image_path, cv::IMREAD_COLOR);

    if (predicted_img.type() != CV_8UC3 || ground_truth_img.type() != CV_8UC3)
    {
        std::cout << "Only 3-channel 8-bit images are supported\n";
        // imread should always return CV_8UC3 matrix, do it just to be sure
        return 1;
    }

    EvaluationResult result = evaluateResult(predicted_img, ground_truth_img);

    std::cout << result.toJson().dump(2);
}