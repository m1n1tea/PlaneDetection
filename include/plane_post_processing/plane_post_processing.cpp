#include <plane_post_processing/plane_post_processing.h>

#include <cstdint>
#include <opencv2/imgproc.hpp>
#include <set>


std::vector<cv::Mat> seperateComponents(cv::Mat mask)
{
    cv::Mat components;
    int components_count = cv::connectedComponents(mask, components, 4) - 1;

    std::vector<cv::Mat> separated_components(components_count);

    for (int i = 0; i < components_count; ++i)
    {
        separated_components[i] = cv::Mat(mask.size(), CV_8U, cv::Scalar(0));
    }

    components.forEach<int32_t>([&separated_components](int32_t& pixel, const int* position) {
        if (pixel != 0)
        {
            separated_components[pixel - 1].at<uint8_t>(position) = 255;
        }
    });

    return separated_components;
}

void cleanNoise(cv::Mat bin_image, int pixel_threshold)
{
    cv::Mat components;
    cv::Mat stats;
    cv::Mat tmp;

    int components_count = cv::connectedComponentsWithStats(bin_image, components, stats, tmp, 4);

    std::vector<std::pair<int, int>> components_areas(components_count);

    for (int i = 0; i < components_count; ++i)
    {
        int area = stats.at<int32_t>(i, cv::CC_STAT_AREA);
        components_areas[i] = {area, i};
    }
    std::sort(components_areas.begin(), components_areas.end());

    int removed_pixels_count = 0;

    std::set<int> removed_components;

    for (int i = 0; i < components_count; ++i)
    {
        removed_pixels_count += components_areas[i].first;
        if (removed_pixels_count <= pixel_threshold)
        {
            removed_components.insert(components_areas[i].second);
        }
        else
        {
            break;
        }
    }

    components.forEach<int32_t>([&removed_components, &bin_image](int32_t& pixel, const int* position) {
        if (removed_components.count(pixel) != 0)
        {
            bin_image.at<uint8_t>(position) = 0;
        }
    });
}

void cleanNoise(cv::Mat bin_image, double relative_threshold)
{
    int pixel_threshold = cv::countNonZero(bin_image) * relative_threshold;
    cleanNoise(bin_image, pixel_threshold);
}

void fillInGaps(cv::Mat bin_image, double relative_threshold, cv::Mat mask)
{
    int pixel_threshold = cv::countNonZero(bin_image) * relative_threshold;

    cv::Mat inv_bin_image;
    cv::bitwise_not(bin_image, inv_bin_image);
    cleanNoise(inv_bin_image, pixel_threshold);
    cv::bitwise_not(inv_bin_image, bin_image);
    cv::bitwise_and(bin_image, mask, bin_image);
}

std::vector<cv::Mat> processPlanes(const std::vector<cv::Mat>& plane_masks, double noise_relative_part,
                                   double gaps_relative_part)
{
    if (plane_masks.empty())
    {
        return {};
    }
    std::vector<cv::Mat> processed_masks;
    cv::Mat mask(plane_masks[0].size(), CV_8U, cv::Scalar(255));
    for (const cv::Mat& bin_image : plane_masks)
    {
        cv::Mat image_copy = bin_image.clone();
        cleanNoise(image_copy, noise_relative_part);
        processed_masks.push_back(image_copy);
        cv::bitwise_xor(mask, image_copy, mask);
    }

    for (cv::Mat& bin_image : processed_masks)
    {
        cv::bitwise_xor(mask, bin_image, mask);
        fillInGaps(bin_image, gaps_relative_part, mask);
        cv::bitwise_xor(mask, bin_image, mask);
    }

    std::vector<cv::Mat> result;

    for (cv::Mat& bin_image : processed_masks)
    {
        std::vector<cv::Mat> components = seperateComponents(bin_image);
        for (cv::Mat& component : components)
        {
            result.push_back(component);
        }
    }
    return result;
}