
#include <focal_length_estimation/focal_length_estimation.h>
#include <plane_detection_utils/plane_detection_utils.h>
#include <plane_labeling/plane_labeling.h>
#include <plane_orientation_detection/plane_orientation_detection.h>
#include <plane_post_processing/plane_post_processing.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace std::chrono;
namespace fs = std::filesystem;

nlohmann::json getDefaultConfig()
{
    nlohmann::json config;
    config["save_intermediate_steps"] = false;
    config["verbose"] = true;

    config["focal_length_estimation"] = nlohmann::json();
    config["focal_length_estimation"]["focal_length"] = 0;
    config["focal_length_estimation"]["adjacency_relative_length_threshold"] = 1.02;
    config["focal_length_estimation"]["adjacency_angle_threshold"] = std::numbers::pi / 18;
    config["focal_length_estimation"]["vanishing_point_relative_threshold"] = 0.03;
    config["focal_length_estimation"]["vanishing_point_absolute_threshold"] = 10;
    config["focal_length_estimation"]["focal_length_mult"] = 1.5;

    config["plane_orientation_detection"] = nlohmann::json();
    config["plane_orientation_detection"]["adjacency_relative_length_threshold"] = 1.25;
    config["plane_orientation_detection"]["adjacency_angle_threshold"] = std::numbers::pi / 18;
    config["plane_orientation_detection"]["ransac_threshold"] = 0.01;
    config["plane_orientation_detection"]["ransac_tries"] = 1000;
    config["plane_orientation_detection"]["plane_relative_threshold"] = 0.1;
    config["plane_orientation_detection"]["plane_absolute_threshold"] = 10;

    config["plane_labeling"] = nlohmann::json();
    config["plane_labeling"]["adjacency_relative_length_threshold"] = 2;
    config["plane_labeling"]["adjacency_angle_threshold"] = std::numbers::pi / 18;

    config["plane_post_processing"] = nlohmann::json();
    config["plane_post_processing"]["noise_relative_part"] = 0.05;
    config["plane_post_processing"]["gaps_relative_part"] = 0.05;
    return config;
}

int main(int argc, char **argv)
{
    const std::string keys =
        "{help h usage ?    |      | print this message }"
        "{@image            |<none>| source image }"
        "{@output_dir       |   .  | directory path to store result }"
        "{c config          |<none>| config file }"
        "{g generate-config |      | generate default config file in the given path}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help") || argc == 1)
    {
        parser.printMessage();
        return 0;
    }
    if (parser.has("generate-config"))
    {
        std::string config_path_str = parser.get<cv::String>("g");
        if (config_path_str == "true")
        {
            config_path_str = "default_config.json";
        }
        fs::path config_path(config_path_str);
        config_path = fs::absolute(config_path);
        std::ofstream config_file;
        if (fs::is_directory(config_path))
        {
            config_path /= "default_config.json";
            config_file.open(config_path.string());
        }
        else if (fs::is_directory(config_path.parent_path()))
        {
            config_file.open(config_path.string());
        }
        else
        {
            std::cout << "invalid generated-config path\n";
            return 1;
        }
        config_file << getDefaultConfig();
        config_file.close();
        return 0;
    }

    nlohmann::json config = getDefaultConfig();
    if (parser.has("config"))
    {
        fs::path config_path(parser.get<cv::String>("c"));
        std::ifstream config_file;
        if (fs::is_regular_file(config_path))
        {
            config_file.open(config_path.string());
        }
        else
        {
            std::cout << "invalid config path\n";
            return 1;
        }
        nlohmann::json tmp_config;
        config_file >> tmp_config;
        for (auto &[key, val] : tmp_config.items())
        {
            if (key == "save_intermediate_steps" || key == "verbose")
            {
                config[key] = val;
            }
            if (key == "focal_length_estimation" || key == "plane_orientation_detection" || key == "plane_labeling" ||
                key == "plane_post_processing")
            {
                for (auto &[subkey, subval] : val.items())
                {
                    config[key][subkey] = subval;
                }
            }
        }
    }

    std::string in_path = parser.get<cv::String>("@image");
    if (!fs::is_regular_file(in_path))
    {
        std::cout << "invalid source image path\n";
        return 1;
    }

    std::string filename = fs::path(in_path).stem().string();

    std::string out_dir_path_str = parser.get<cv::String>("@output_dir");
    fs::path out_dir_path(out_dir_path_str);
    out_dir_path = fs::absolute(out_dir_path);
    if (!fs::is_directory(out_dir_path))
    {
        std::cout << "invalid output directory path\n";
        return 1;
    }
    cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);

    auto start = high_resolution_clock::now();

    if (img.type() != CV_8UC3)
    {
        std::cout << "Only 3-channel 8-bit images are supported\n";
        // imread should always return CV_8UC3 matrix, do it just to be sure
        return 1;
    }

    std::vector<LineSegment> lines = detectLines(img);

    auto end1 = high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end1 - start;

    if (config["verbose"])
    {
        std::cout << "found " << lines.size() << " lines" << std::endl;
        std::cout << "line search took: " << duration1.count() << " seconds" << std::endl;
    }

    double f = config["focal_length_estimation"]["focal_length"];
    if (f <= 0)
    {
        f = findFocalLength(lines, config["focal_length_estimation"]["adjacency_relative_length_threshold"],
                            config["focal_length_estimation"]["adjacency_angle_threshold"],
                            config["focal_length_estimation"]["vanishing_point_relative_threshold"],
                            config["focal_length_estimation"]["vanishing_point_absolute_threshold"]);
        f *= config["focal_length_estimation"]["focal_length_mult"];
    }
    if (f == 0)
    {
        f = std::max(img.rows, img.cols);
    }
    auto end2 = high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end2 - end1;
    if (config["verbose"])
    {
        std::cout << "estimated focal length: " << f << std::endl;
        std::cout << "focal length estimation took: " << duration2.count() << " seconds" << std::endl;
    }
    std::vector<PlaneInfo> planes =
        getPlanes(lines, f, config["plane_orientation_detection"]["ransac_threshold"],
                  config["plane_orientation_detection"]["ransac_tries"],
                  config["plane_orientation_detection"]["adjacency_relative_length_threshold"],
                  config["plane_orientation_detection"]["adjacency_angle_threshold"],
                  config["plane_orientation_detection"]["plane_relative_threshold"],
                  config["plane_orientation_detection"]["plane_absolute_threshold"]);

    auto end3 = high_resolution_clock::now();
    std::chrono::duration<double> duration3 = end3 - end2;
    if (config["verbose"])
    {
        std::cout << "found " << planes.size() << " plane orientations" << std::endl;
        std::cout << "plane orientations angles:\n";
        for (const PlaneInfo &plane : planes)
        {
            std::cout << "    " << plane.a << " " << plane.b << std::endl;
        }
        std::cout << "plane orientation detection took: " << duration3.count() << " seconds" << std::endl;
    }

    distributeLinesBeetweenPlanes(planes, lines, f, config["plane_orientation_detection"]["ransac_threshold"],
                                  config["plane_labeling"]["adjacency_relative_length_threshold"],
                                  config["plane_labeling"]["adjacency_angle_threshold"]);
    std::vector<cv::Mat> masks = findPlaneRegions(planes, lines, f, img.size());
    std::vector<cv::Mat> processed_masks = processPlanes(masks, config["plane_post_processing"]["noise_relative_part"],
                                                         config["plane_post_processing"]["gaps_relative_part"]);

    auto end4 = high_resolution_clock::now();
    std::chrono::duration<double> duration4 = end4 - end3;

    if (config["verbose"])
    {
        std::cout << "found " << processed_masks.size() << " plane regions" << std::endl;
        std::cout << "plane labeling took: " << duration4.count() << " seconds" << std::endl;
    }

    std::vector<cv::Scalar> planeColors;

    for (int i = 0; i < processed_masks.size(); ++i)
    {
        cv::Scalar line_color;
        randu(line_color, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        planeColors.push_back(line_color);
    }

    if (config["save_intermediate_steps"])
    {
        cv::Mat img_clone1 = img.clone();
        for (const LineSegment &line : lines)
        {
            cv::line(img_clone1, line.start, line.end, cv::Scalar(), 4);
        }
        cv::Mat img_clone2 = img.clone();
        for (int i = 0; i < lines.size(); ++i)
        {
            for (int p = 0; p < planes.size(); ++p)
            {
                if (!planes[p].adjacency_graph[i].empty())
                {
                    cv::line(img_clone2, lines[i].start, lines[i].end, planeColors[p], 4);
                }
            }
        }

        cv::Mat img_clone3 = img.clone();
        for (int p = 0; p < planes.size(); ++p)
        {
            cv::Mat plane_color_mat(img.size(), img.type(), planeColors[p]);
            cv::Mat dst;
            cv::addWeighted(img_clone3, 0.5, plane_color_mat, 0.5, 0.0, dst);
            cv::copyTo(dst, img_clone3, masks[p]);
        }

        cv::Mat img_clone4 = img.clone();

        for (int p = 0; p < processed_masks.size(); ++p)
        {
            cv::Mat plane_color_mat(img.size(), img.type(), planeColors[p]);
            cv::Mat dst;
            cv::addWeighted(img_clone4, 0.5, plane_color_mat, 0.5, 0.0, dst);
            cv::copyTo(dst, img_clone4, processed_masks[p]);
        }

        std::string out_path1 = (out_dir_path / (filename + "_detected_lines.png")).string();
        std::string out_path2 = (out_dir_path / (filename + "_detected_labeled_lines.png")).string();
        std::string out_path3 = (out_dir_path / (filename + "_detected_labeled_pixels.png")).string();
        std::string out_path4 = (out_dir_path / (filename + "_detected_labeled_pixels_processed.png")).string();
        cv::imwrite(out_path1, img_clone1);
        cv::imwrite(out_path2, img_clone2);
        cv::imwrite(out_path3, img_clone3);
        cv::imwrite(out_path4, img_clone4);
    }

    std::string out_path = (out_dir_path / (filename + "_result.png")).string();
    cv::Mat result(img.size(), CV_8UC3, cv::Scalar());
    for (int p = 0; p < processed_masks.size(); ++p)
    {
        cv::Mat plane_color_mat(img.size(), CV_8UC3, planeColors[p]);
        cv::copyTo(plane_color_mat, result, processed_masks[p]);
    }
    cv::imwrite(out_path, result);
}