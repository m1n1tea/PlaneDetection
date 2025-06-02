#include "semcv.h"
#include <fstream>
#include <cmath>
#include <limits>
#include <numbers>
#include <algorithm>
#include <memory>
#include <random>
#include <VPCluster.h>
#include <VPSample.h>


namespace{
    cv::Point2d inf = {std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};

    std::random_device rd;   // non-deterministic generator
}


LineSegment::LineSegment(cv::Point2d start,cv::Point2d end):start(start),end(end),vp_id(-1){}
LineSegment::LineSegment(double x1, double y1, double x2, double y2):start(x1,y1),end(x2,y2),vp_id(-1){}
LineSegment::LineSegment(cv::Vec4d vec):start(vec[0],vec[1]),end(vec[2],vec[3]),vp_id(-1){}

cv::Vec3d LineSegment::getLineHomogeneousCoordinates() const{
    cv::Vec3d p1 = {start.x,start.y,1};
    cv::Vec3d p2 = {end.x,end.y,1};
    return p1.cross(p2);
}

cv::Point2d LineSegment::getVector() const{
    return end-start;
}

cv::Point2d getIntersection(const LineSegment& lhs, const LineSegment& rhs){
    cv::Vec3d hom_point = lhs.getLineHomogeneousCoordinates().cross(rhs.getLineHomogeneousCoordinates());
    if(hom_point[2]==0){
        return inf;
    }
    hom_point/=hom_point[2];
    return {hom_point[0],hom_point[1]};
}

bool areAdjacent(const LineSegment& lhs, const LineSegment& rhs){

    double cos_threshold = std::cos(std::numbers::pi/18);
    double relative_dist_threshold = 1.1;

    cv::Point2d vec1 = lhs.getVector();
    cv::Vec2d vec2 = rhs.getVector();
    double line_len1 = norm(vec1);
    double line_len2 = norm(vec2);
    vec1/=line_len1;
    vec2/=line_len2;
    double cos = std::abs(vec1.ddot(vec2));
    if(cos>cos_threshold){
        return false;
    }
    cv::Point2d int_point = getIntersection(lhs,rhs);
    if(int_point == inf){
        return false;
    }
    double line_with_intersection_len1 = std::max(norm(lhs.end-int_point),norm(lhs.start-int_point));
    double line_with_intersection_len2 = std::max(norm(rhs.end-int_point),norm(rhs.start-int_point));

    if(line_with_intersection_len1/line_len1 > relative_dist_threshold ||  line_with_intersection_len2/line_len2 > relative_dist_threshold){
        return false;
    }
    return true;

}





std::vector<cv::Point3d> findVanishingPoints(std::vector<LineSegment>& lines){
    	std::vector< std::vector<float> *> pts;

        for(size_t i=0;i<lines.size();++i){
            std::vector<float>* p = new std::vector<float>(4);
            (*p)[0]=lines[i].start.x;
            (*p)[1]=lines[i].start.y;
            (*p)[2]=lines[i].end.x;
            (*p)[3]=lines[i].end.y;
            pts.push_back(p);
        }

		std::vector<unsigned int> Labels;
		std::vector<unsigned int> LabelCount;
		std::vector<unsigned int> modelIndex;
		std::vector<std::vector<float> *> *mModels = 
		VPSample::run(&pts, 5000, 2, 0, 3);
		int classNum = VPCluster::run(Labels, LabelCount, modelIndex, &pts, mModels, 1.0, 2);

        std::vector<cv::Point3d> vps(modelIndex.size());

        for(size_t i=0;i<modelIndex.size();++i){
            const std::vector<float>& vp = *((*mModels)[modelIndex[i]]);
            vps[i]=cv::Point3d(vp[0],vp[1],vp[2]);
        }
        for(size_t i=0;i<Labels.size();++i){
            lines[i].vp_id=Labels[i];
        }
        for(size_t i=0; i < mModels->size(); ++i)
			delete (*mModels)[i];
		delete mModels;
		for(size_t i=0; i<pts.size(); i++)
			delete pts[i];
        
            return vps;
}

double estimateFocalLength(const std::vector<cv::Point3d>& vanishing_points,const std::vector<std::vector<uint32_t>>& orth_mat){
    double num_sum=0;
    double denom_sum=0;

    for(size_t i=0;i<vanishing_points.size();++i){
        for(size_t j=0;j<i;++j){
            if(orth_mat[i][j]==0){
                continue;
            }
            const cv::Point3d& vpi = vanishing_points[i];
            const cv::Point3d& vpj = vanishing_points[j];

            num_sum+=vpi.z*vpj.z*(vpi.x*vpj.x+vpi.y*vpj.y)/(norm(vpi)*norm(vpj));
            denom_sum+=vpi.z*vpj.z*vpi.z*vpj.z/(norm(vpi)*norm(vpj));
        }
    }
    return std::sqrt(-num_sum/denom_sum);
}


std::vector<std::vector<int>> findAdjacencyGraph(std::vector<LineSegment>& lines){
    int n = lines.size();
    std::vector<std::vector<int>> graph(n);
    for(int i=0;i<n;++i){
        for(int j=0;j<i;++j){
            if(areAdjacent(lines[i],lines[j])){
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }
    return graph;
}

bool possibleOrth(cv::Point3d vpi, cv::Point3d vpj){
    return vpi.z*vpj.z*(vpi.x*vpj.x+vpi.y*vpj.y) < 0;
}

std::vector<std::vector<uint32_t>> findOrthMat(const std::vector<std::vector<int>>& adjacency_graph, const std::vector<LineSegment>& lines, const std::vector<cv::Point3d>& vanishing_points){

    std::vector<std::vector<uint32_t>> mat(vanishing_points.size(),std::vector<uint32_t>(vanishing_points.size()));
    int sum=0;
    uint32_t max_val = 0;

    double relative_threshold = 0.1;
    double absolute_threshold = 10;

    for(size_t i=0;i<adjacency_graph.size();++i){
        for(int j : adjacency_graph[i]){
            unsigned int vp_id1 = lines[i].vp_id;
            unsigned int vp_id2 = lines[j].vp_id;
            if(!possibleOrth(vanishing_points[vp_id1],vanishing_points[vp_id2])){
                continue;
            }
            sum+=1;
            mat[vp_id1][vp_id2]+=1;
            mat[vp_id2][vp_id1]+=1;
            max_val = std::max(mat[vp_id1][vp_id2],max_val);
        }
    }
    for(int i=0;i<vanishing_points.size();++i){
        for(int j=0;j<vanishing_points.size();++j){
            if(mat[i][j]==max_val || (mat[i][j]>=absolute_threshold && ((double)mat[i][j])/sum >= relative_threshold)){
                mat[i][j]=1;
            }
            else{
                mat[i][j]=0;
            }
        }
    }

    return mat;
}



casadi::MX xRotationMat(const casadi::MX& a){

    casadi::MX cos_a = casadi::MX::cos(a);
    casadi::MX sin_a = casadi::MX::sin(a);
    
    casadi::MX rotation_mat = casadi::MX::vertcat(
        std::vector<casadi::MX>
        {
            casadi::MX::horzcat(std::vector<casadi::MX>{1, 0, 0}),
            casadi::MX::horzcat(std::vector<casadi::MX>{0, cos_a, -sin_a}),
            casadi::MX::horzcat(std::vector<casadi::MX>{0, sin_a, cos_a})
        }
    );
    return rotation_mat;
}

casadi::MX yRotationMat(const casadi::MX& b){

    casadi::MX cos_b = casadi::MX::cos(b);
    casadi::MX sin_b = casadi::MX::sin(b);
    
    casadi::MX rotation_mat = casadi::MX::vertcat(
    {
        casadi::MX::horzcat({cos_b, 0, -sin_b}),
        casadi::MX::horzcat({0, 1, 0}),
        casadi::MX::horzcat({sin_b, 0, cos_b})
    });
    return rotation_mat;
}
casadi::MX transformationMatInvT(const casadi::MX& a,const casadi::MX& b,const casadi::MX& f){
    casadi::MX calibration_mat = casadi::MX::vertcat(
    {
        casadi::MX::horzcat({f, 0, 0}),
        casadi::MX::horzcat({0, f, 0}),
        casadi::MX::horzcat({0, 0, 1})
    });
    casadi::MX result = casadi::MX::mtimes(xRotationMat(a),yRotationMat(b));
    return casadi::MX::mtimes(result,calibration_mat);
}



std::pair<double,double> find_ab(const LineSegment& segment_i,const LineSegment& segment_j, double f_val){

    casadi::Opti opti;
    opti.solver("ipopt");
    casadi::MX li = opti.parameter(3);
    casadi::MX lj = opti.parameter(3);
    casadi::MX f = opti.parameter(1);
    cv::Vec3d li_coords = segment_i.getLineHomogeneousCoordinates();
    cv::Vec3d lj_coords = segment_j.getLineHomogeneousCoordinates();
    opti.set_value(li,{li_coords[0],li_coords[1],li_coords[2]});
    opti.set_value(lj,{lj_coords[0],lj_coords[1],lj_coords[2]});
    opti.set_value(f,f_val);
    casadi::MX a = opti.variable();
    casadi::MX b = opti.variable();

    casadi::MX H = transformationMatInvT(a,b,f);


    double eps = 1e-10;

    casadi::MX vi = (casadi::MX::mtimes(H,li));
    casadi::MX vj = (casadi::MX::mtimes(H,lj));

    opti.minimize((casadi::MX::dot(vi,vj)*casadi::MX::dot(vi,vj))/(casadi::MX::dot(vi,vi)*casadi::MX::dot(vj,vj)+eps));

    opti.solve();

    double a_val = static_cast<double>(opti.value(a));
    double b_val = static_cast<double>(opti.value(b));
    return {a_val,b_val};
}

double calcOrthogonality(const LineSegment& segment_i,const LineSegment& segment_j, double f_val, double a_val, double b_val){
    casadi::MX a = casadi::MX::sym("a");
    casadi::MX b = casadi::MX::sym("b");
    casadi::MX f = casadi::MX::sym("f");;
    casadi::MX li = casadi::MX::sym("li",3);
    casadi::MX lj = casadi::MX::sym("lj",3);

    casadi::MX H = transformationMatInvT(a,b,f);

    casadi::MX vi = (casadi::MX::mtimes(H,li));
    casadi::MX vj = (casadi::MX::mtimes(H,lj));

    double eps = 1e-10;

    vi/=(casadi::MX::norm_2(vi)+eps);
    vj/=(casadi::MX::norm_2(vj)+eps);
    casadi::MX func_out = (casadi::MX::dot(vi,vj)*casadi::MX::dot(vi,vj))/(casadi::MX::dot(vi,vi)*casadi::MX::dot(vj,vj)+eps);

    casadi::Function func("func", {a,b,f,li,lj},{func_out});

    cv::Vec3d li_coords = segment_i.getLineHomogeneousCoordinates();
    cv::Vec3d lj_coords = segment_j.getLineHomogeneousCoordinates();

    std::vector<casadi::DM> res = func(std::vector<casadi::DM>{a_val,b_val,f_val,{li_coords[0],li_coords[1],li_coords[2]},{lj_coords[0],lj_coords[1],lj_coords[2]}});

    return static_cast<double>(res[0]);
}


int count_nonzeros(const std::vector<uint8_t>& mask){
    int sum=0;
    for(uint8_t el : mask){
        if (el!=0){
            sum+=1;
        }
    }
    return sum;
}


std::pair<double,double> findDominantPlane(std::vector<uint8_t>& inliners, const std::vector<LineSegment>& lines, const std::vector<std::vector<int>>& adjacency_graph, double f, double t, int N){
    int n = lines.size();
    int edges_count = 0;

    for(int i=0;i<n;++i){
        edges_count+=adjacency_graph[i].size();
    }

    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0,edges_count-1);

    std::pair<double,double> best_ab;

    for(int o=0;o<N;++o){
        int edge_num = dist(gen);
        int v1=-1;
        int v2=-1;
        for(int i=0;i<n;++i){
            if(edge_num>=adjacency_graph[i].size()){
                edge_num-=adjacency_graph[i].size();
            }
            else{
                v1=i;
                v2=edge_num;
                break;
            }
        }
        std::pair<double,double> ab = find_ab(lines[v1],lines[v2],f);
        std::vector<uint8_t> local_inliners(n,0);
        for(int i=0;i<n;++i){
            for(int j : adjacency_graph[i]){
                if(local_inliners[i]!=0 && local_inliners[j]!=0 ){
                    continue;
                }
                if (calcOrthogonality(lines[i],lines[j],f,ab.first,ab.second)<t){
                    local_inliners[i]=1;
                    local_inliners[j]=1;
                }
            }
        }
        if(count_nonzeros(local_inliners)>count_nonzeros(inliners)){
            inliners=std::move(local_inliners);
            best_ab=ab;
        }
    }
    return best_ab;
}

void remove_lines(std::vector<LineSegment>& lines, std::vector<std::vector<int>>& adjacency_graph, const std::vector<uint8_t>& mask, bool remove_zeros){
    int n = lines.size();
    std::vector<int> transformation(n,-1);
    int new_i = 0;
    for(int i=0;i<n;++i){
        if((mask[i]==0) == remove_zeros){
            //remove this element
            continue;
        }
        transformation[i]=new_i;
        lines[new_i]=std::move(lines[i]);
        adjacency_graph[new_i]=std::move(adjacency_graph[i]);
        new_i++;
    }
    n = new_i;
    for(int i=0;i<n;++i){
        for(int& j : adjacency_graph[i]){
            j = transformation[j];
        }
    }
    lines.resize(n);
    adjacency_graph.resize(n);
}

std::vector<PlaneInfo> getPlanes(const std::vector<LineSegment>& lines, const std::vector<std::vector<int>>& adjacency_graph, double f, double t, int N){
    int n = lines.size();
    auto lines_copy = lines;
    auto adjacency_graph_copy = adjacency_graph;

    int pairs_count = 0;
    for(int i=0;i<n;++i){
        pairs_count+=adjacency_graph[i].size();
    }
    double relative_threshold = 0.1;
    double absolute_threshold = 10;

    std::vector<PlaneInfo> planes;

    do{
        PlaneInfo plane;
        std::vector<uint8_t> inliners;
        std::pair<double,double> ab = findDominantPlane(inliners, lines_copy, adjacency_graph_copy, f, t, N);
        plane.a=ab.first;
        plane.b=ab.second;
        plane.inliners=lines_copy;
        plane.adjacency_graph=adjacency_graph_copy;
        remove_lines(lines_copy, adjacency_graph_copy, inliners, false);
        remove_lines(plane.inliners, plane.adjacency_graph, inliners, true);

        int plane_pairs_count = 0;
        for(int i=0;i<plane.adjacency_graph.size();++i){
            pairs_count+=plane.adjacency_graph[i].size();
        }
        if(plane_pairs_count>=absolute_threshold && ((double)plane_pairs_count)/pairs_count >= relative_threshold){
            planes.push_back(plane);
        }
        else{
            break;
        }
        
    }while(true);

    return planes;
}
