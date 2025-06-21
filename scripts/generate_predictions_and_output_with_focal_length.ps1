
$images_path = "$PSScriptRoot\..\images"
$ground_truth_path = "$PSScriptRoot\..\ground_truth"
$generated_images_path = "$PSScriptRoot\..\generated_images"
$generated_reports_path = "$PSScriptRoot\..\generated_reports"

$plane_detector_path = "$PSScriptRoot\..\bin\plane_detector.exe"
$compare_planes_path = "$PSScriptRoot\..\bin\compare_planes.exe"

mkdir "$generated_images_path" -ea 0
mkdir "$generated_reports_path" -ea 0

$input = @("Image_1.jpg","Image_2.jpg","Image_3.jpg","multi_plane_non_orth1.jpg","multi_plane_non_orth2.jpg","multi_plane_orth.jpg","sample_image.jpg","strange_building.jpg","tall_building.png","wide_building.jpg")
$result = @("Image_1_result.png","Image_2_result.png","Image_3_result.png","multi_plane_non_orth1_result.png","multi_plane_non_orth2_result.png","multi_plane_orth_result.png","sample_image_result.png","strange_building_result.png","tall_building_result.png","wide_building_result.png")
$ground_truth = @("Image_1_ground_truth.png","Image_2_ground_truth.png","Image_3_ground_truth.png","multi_plane_non_orth1_ground_truth.png","multi_plane_non_orth2_ground_truth.png","multi_plane_orth_ground_truth.png","sample_image_ground_truth.png","strange_building_ground_truth.png","tall_building_ground_truth.png","wide_building_ground_truth.png")
$report = @("Image_1__report.json","Image_2__report.json","Image_3__report.json","multi_plane_non_orth1__report.json","multi_plane_non_orth2__report.json","multi_plane_orth__report.json","sample_image__report.json","strange_building__report.json","tall_building__report.json","wide_building__report.json")
$focal_length = 1500, 1500, 450, 860, 600, 2500, 1600, 1100, 1300, 1600

$a = 0..9
foreach ($element in $a) {
  Write-Output $($input[$element])
  & "$plane_detector_path" -f="$($focal_length[$element])" $images_path\$($input[$element]) $generated_images_path
}

foreach ($element in $a) {
  & "$compare_planes_path" $generated_images_path\$($result[$element]) $ground_truth_path\$($ground_truth[$element]) | Out-File -FilePath $generated_reports_path\$($report[$element])
  Write-Output $($input[$element])
  Get-Content $generated_reports_path\$($report[$element]) | Select-String -Pattern "f1_plane_separation"
}


