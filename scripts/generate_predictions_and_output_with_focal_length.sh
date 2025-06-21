#!/bin/bash -e

MY_PATH="$(dirname -- "${BASH_SOURCE[0]}")"

images_path="$MY_PATH/../images"
ground_truth_path="$MY_PATH/../ground_truth"
generated_images_path="$MY_PATH/../generated_images"
generated_reports_path="$MY_PATH/../generated_reports"

plane_detector_path="$MY_PATH/../bin/plane_detector"
compare_planes_path="$MY_PATH/../bin/compare_planes"

mkdir -p "$generated_images_path"
mkdir -p "$generated_reports_path"

input=("Image_1.jpg" "Image_2.jpg" "Image_3.jpg" "multi_plane_non_orth1.jpg" "multi_plane_non_orth2.jpg" "multi_plane_orth.jpg" "sample_image.jpg" "strange_building.jpg" "tall_building.png" "wide_building.jpg")
result=("Image_1_result.png" "Image_2_result.png" "Image_3_result.png" "multi_plane_non_orth1_result.png" "multi_plane_non_orth2_result.png" "multi_plane_orth_result.png" "sample_image_result.png" "strange_building_result.png" "tall_building_result.png" "wide_building_result.png")
ground_truth=("Image_1_ground_truth.png" "Image_2_ground_truth.png" "Image_3_ground_truth.png" "multi_plane_non_orth1_ground_truth.png" "multi_plane_non_orth2_ground_truth.png" "multi_plane_orth_ground_truth.png" "sample_image_ground_truth.png" "strange_building_ground_truth.png" "tall_building_ground_truth.png" "wide_building_ground_truth.png")
report=("Image_1_ground_report.json" "Image_2_ground_report.json" "Image_3_ground_report.json" "multi_plane_non_orth1_ground_report.json" "multi_plane_non_orth2_ground_report.json" "multi_plane_orth_ground_report.json" "sample_image_ground_report.json" "strange_building_ground_report.json" "tall_building_ground_report.json" "wide_building_ground_report.json")
focal_length=(1500 1500 450 860 600 2500 1600 1100 1300 1600)


for element in {0..9}
do
  echo "${input[$element]}"
  $plane_detector_path -f="${focal_length[$element]}" $images_path/${input[$element]} $generated_images_path
done

for element in {0..9}
do
  echo "${input[$element]}"
  $compare_planes_path $generated_images_path/${result[$element]} $ground_truth_path/${ground_truth[$element]} > $generated_reports_path/${report[$element]}
  grep "f1_plane_separation" $generated_reports_path/${report[$element]}
done


