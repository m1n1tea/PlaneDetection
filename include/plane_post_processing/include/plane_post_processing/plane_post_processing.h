

#ifndef PLANE_POST_PROCESSING
#define PLANE_POST_PROCESSING

#include <opencv2/core.hpp>
#include <vector>

std::vector<cv::Mat> seperateComponents(cv::Mat mask);

void cleanNoise(cv::Mat bin_image, int pixel_threshold);

void cleanNoise(cv::Mat bin_image, double relative_threshold);
void fillInGaps(cv::Mat bin_image, double relative_threshold, cv::Mat mask);

std::vector<cv::Mat> processPlanes(const std::vector<cv::Mat>& plane_masks, double noise_relative_part,
                                   double gaps_relative_part);

#endif  // PLANE_LABELING