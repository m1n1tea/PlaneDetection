
$images_path = "$PSScriptRoot\..\images"
$ground_truth_path = "$PSScriptRoot\..\ground_truth"
$generated_images_path = "$PSScriptRoot\..\generated_images"
$generated_reports_path = "$PSScriptRoot\..\generated_reports"

mkdir "$generated_images_path" -ea 0
mkdir "$generated_reports_path" -ea 0

$input = @("Image_1.jpg","Image_2.jpg","Image_3.jpg","multi_plane_non_orth1.jpg","multi_plane_non_orth2.jpg","multi_plane_orth.jpg","sample_image.jpg","strange_building.jpg","tall_building.png","wide_building.jpg")
$result = @("Image_1_result.png","Image_2_result.png","Image_3_result.png","multi_plane_non_orth1_result.png","multi_plane_non_orth2_result.png","multi_plane_orth_result.png","sample_image_result.png","strange_building_result.png","tall_building_result.png","wide_building_result.png")
$ground_truth = @("Image_1_ground_truth.png","Image_2_ground_truth.png","Image_3_ground_truth.png","multi_plane_non_orth1_ground_truth.png","multi_plane_non_orth2_ground_truth.png","multi_plane_orth_ground_truth.png","sample_image_ground_truth.png","strange_building_ground_truth.png","tall_building_ground_truth.png","wide_building_ground_truth.png")
$report = @("Image_1_ground_report.json","Image_2_ground_report.json","Image_3_ground_report.json","multi_plane_non_orth1_ground_report.json","multi_plane_non_orth2_ground_report.json","multi_plane_orth_ground_report.json","sample_image_ground_report.json","strange_building_ground_report.json","tall_building_ground_report.json","wide_building_ground_report.json")

$a = 0..9
foreach ($element in $a) {
  Write-Output $($input[$element])
  & "$PSScriptRoot\plane_detector.exe" $images_path\$($input[$element]) $generated_images_path
}

foreach ($element in $a) {
  & "$PSScriptRoot\compare_planes.exe" $generated_images_path\$($result[$element]) $ground_truth_path\$($ground_truth[$element]) | Out-File -FilePath $generated_reports_path\$($report[$element])
  Write-Output $($input[$element])
  Get-Content $generated_reports_path\$($report[$element]) | Select-String -Pattern "f1_plane_separation"
}


