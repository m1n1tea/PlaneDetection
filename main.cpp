

#include <semcv.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include<iostream>
#include <fstream>
#include <chrono>
#include <casadi/casadi.hpp>
using namespace std::chrono;



int main(){
    casadi::Importer::load_plugin("ipopt");
    // std::cout << "IPOPT available: " << casadi::Importer::has_plugin("ipopt") << std::endl;

    auto start = high_resolution_clock::now();

    std::string in_path = R"(C:\Users\sirma\git_repos\hough_transform\big_buildings.png)";
    std::string out_path1 = R"(C:\Users\sirma\git_repos\hough_transform\sample_lines.txt)";
    std::string out_path2 = R"(C:\Users\sirma\git_repos\hough_transform\sample_planes.txt)";
    cv::Mat img = cv::imread(in_path);
    
    auto detector = cv::line_descriptor::LSDDetector::createLSDDetector();

    std::vector<cv::line_descriptor::KeyLine> keylines;

    detector->detect(img, keylines, 1.2, 1, cv::Mat());
    int n = keylines.size();
    std::vector<LineSegment> lines;
    lines.reserve(n);

    for(size_t i = 0; i<n ; i++)
    {
        lines.emplace_back(keylines[i].startPointX,keylines[i].startPointY,keylines[i].endPointX,keylines[i].endPointY);
    }

    cv::Mat img_clone = img.clone();
    for(const LineSegment& line : lines ){
        cv::line(img_clone,line.start,line.end,cv::Scalar(0,0,0),2);
    }

    std::vector<std::vector<int>> adjacency_graph = findAdjacencyGraph(lines);

    std::vector<uint8_t> singleLines(n,0);
    for(int i=0; i<n;++i){
        singleLines[i]=adjacency_graph[i].empty();
    }
    remove_lines(lines,adjacency_graph,singleLines,false);

    n=lines.size();

    auto end1 = high_resolution_clock::now();
    std::vector<cv::Point3d> vps = findVanishingPoints(lines);




    auto end2 = high_resolution_clock::now();
    std::vector<std::vector<uint32_t>> orth_mat = findOrthMat(adjacency_graph, lines, vps);

    double f = estimateFocalLength(vps,orth_mat);

    std::vector<PlaneInfo> planes = getPlanes(lines,adjacency_graph, f);

    for(const PlaneInfo& plane : planes){
        cv::Scalar line_color;
        randu(line_color, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        for(const LineSegment& line : plane.inliners){
            cv::line(img,line.start,line.end,line_color,3);
        }
    }
    auto end3 = high_resolution_clock::now();

    auto duration1 = duration_cast<microseconds>(end1 - start);
    auto duration2 = duration_cast<microseconds>(end2 - start);
    auto duration3 = duration_cast<microseconds>(end3 - start);

    std::cout << duration1.count() << std::endl;
    std::cout << duration2.count() << std::endl;
    std::cout << duration3.count() << std::endl;

    cv::imshow("predicted planes",img);
    cv::imshow("detected_lines",img_clone);

    cv::imwrite(out_path1,img_clone);
    cv::imwrite(out_path2,img);
    cv::waitKey();
}