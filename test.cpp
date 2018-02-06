#include <iostream>
using namespace std;

// python code/fit_3d.py $PWD --viz
// cd code;
// python smpl_webuser/hello_world/render_smpl.py

#include <renderer.hpp>
#include <eigen3/Eigen/Eigen>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/reader.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iomanip>
#include <chrono>

class SMPL{
public:
    Eigen::MatrixXf mV, mVTemp;
    Eigen::MatrixXf mF;
    Eigen::MatrixXf mPose;
    Eigen::MatrixXf mKintreeTable;
    Eigen::MatrixXf mJ;
    Eigen::MatrixXf mTrans;
    Eigen::MatrixXf mWeights;
    Eigen::MatrixXf mWeightsT;
    Eigen::MatrixXf vertSymIdxs;
    typedef Eigen::Matrix<float, 4, 24> BlockMatrix;
    std::vector<Eigen::MatrixXf> weightedBlockMatrix1;
    std::vector<BlockMatrix> blocks;

    SMPL(){
        blocks.resize(4);
        weightedBlockMatrix1.resize(4);
    }

    void printEigen(const Eigen::MatrixXf& m, int colCount = 4, int rowCount = 4){
        std::cout << std::setprecision(3) << std::fixed;
        cout << "R: " << m.rows() << " C: " << m.cols() << endl;
        cout << "[";
        for(int i=0; i<m.rows(); i++){
            if(i < rowCount){
                cout << "[";
                for(int j=0; j<m.cols(); j++){
                    if(j < colCount)
                        cout << m(i,j) << "     ";
                    else if(j == colCount)
                        cout << " ... ";
                    else if(j >= m.cols()-colCount)
                        cout << m(i,j) << "     ";
                }
                cout << "]" << endl;
            }else if(i == rowCount){
                cout << "..." << endl;
                cout << "..." << endl;
            }else if(i >= m.rows()-rowCount){
                cout << "[";
                for(int j=0; j<m.cols(); j++){
                    if(j < colCount)
                        cout << m(i,j) << "     ";
                    else if(j == colCount)
                        cout << " ... ";
                    else if(j >= m.cols()-colCount)
                        cout << m(i,j) << "     ";
                }
                cout << "]" << endl;
            }
        }
        cout << "]" << endl;
    }

    bool loadEigenFromJSON(const Json::Value& json, Eigen::MatrixXf& m){
        // Set Shape
        int cols = json.size();
        if(!cols) { cerr << "Matrix Has no Cols" << endl; return false;}
        int rows = json[0].size();
        if(rows == 0) rows = 1;
        m.resize(rows, cols);

        // Load Data
        for(int i=0; i<cols; i++){
            if(rows > 1){
                for(int j=0; j<rows; j++){
                    m(j,i) = json[i][j].asFloat();
                }
            }else{
                m(0,i) = json[i].asFloat();
            }
        }

        Eigen::MatrixXf t;
        t = m.transpose();
        m = t;

        return true;
    }

    bool loadModelFromJSONFile(std::string filePath){
        ifstream in(filePath);
        Json::Value root;
        in >> root;
        if(!root.size()){
            cerr << "Failed to load model file" << endl;
            return false;
        }

        cout << root["pose_training_info"] << endl;

        loadEigenFromJSON(root["pose"], mPose);
        mPose = Eigen::Map<Eigen::MatrixXf>(mPose.data(), 24,3); // row-col

        loadEigenFromJSON(root["f"], mF);

        loadEigenFromJSON(root["kintree_table"], mKintreeTable);

        loadEigenFromJSON(root["J"], mJ);

        loadEigenFromJSON(root["trans"], mTrans);

        loadEigenFromJSON(root["v_posed"], mV);
        mV.conservativeResize(mV.rows(),mV.cols()+1);
        mV.col(mV.cols()-1) = Eigen::VectorXf::Ones(mV.rows());
        mVTemp = mV;
        for(Eigen::MatrixXf& w : weightedBlockMatrix1) w.resize(4,mV.rows());

        loadEigenFromJSON(root["weights"], mWeights);
        mWeightsT = mWeights.transpose();

        loadEigenFromJSON(root["vert_sym_idxs"], vertSymIdxs);

        return true;
    }

    Eigen::Matrix4f rod(const Eigen::Vector3f& v, const Eigen::Vector3f& t){
        Eigen::Matrix4f m;
        cv::Mat src(cv::Size(1,3),CV_32FC1,cv::Scalar(0));
        src.at<float>(0) = v(0);
        src.at<float>(1) = v(1);
        src.at<float>(2) = v(2);
        cv::Mat dst;
        cv::Rodrigues(src, dst);
        m(0,0) = dst.at<float>(0,0);
        m(0,1) = dst.at<float>(0,1);
        m(0,2) = dst.at<float>(0,2);
        m(0,3) = t(0);
        m(1,0) = dst.at<float>(1,0);
        m(1,1) = dst.at<float>(1,1);
        m(1,2) = dst.at<float>(1,2);
        m(1,3) = t(1);
        m(2,0) = dst.at<float>(2,0);
        m(2,1) = dst.at<float>(2,1);
        m(2,2) = dst.at<float>(2,2);
        m(2,3) = t(2);
        m(3,0) = 0;
        m(3,1) = 0;
        m(3,2) = 0;
        m(3,3) = 1;
        return m;
    }

    bool updateModel(){

        // Create parent link table
        // {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
        std::map<int,int> parent;
        for(int i=1; i<mKintreeTable.cols(); i++){
            int key = mKintreeTable(1,i); int val = mKintreeTable(0,i);
            parent[key] = val;
        }

        std::vector<Eigen::Matrix4f> globalTransforms(24);
        std::vector<Eigen::Matrix4f> transforms(24);

        // Body pose
        Eigen::Matrix4f& bodyPose = globalTransforms[0];
        bodyPose = rod(mPose.row(0), mJ.row(0));

        // Global Transforms
        for(int i=1; i<globalTransforms.size(); i++){
            Eigen::Matrix4f& pose = globalTransforms[i];
            pose = globalTransforms[parent[i]] * rod(mPose.row(i), mJ.row(i) - mJ.row(parent[i]));
        }

        // Transforms
        for(int i=0; i<transforms.size(); i++){
            Eigen::Matrix4f& pose = transforms[i];
            Eigen::Vector4f jZero;
            jZero << mJ(i,0), mJ(i,1), mJ(i,2), 0;
            Eigen::Vector4f fx = globalTransforms[i] * jZero; // Only apply rot to jVector
            Eigen::Matrix4f pack = Eigen::Matrix4f::Zero();
            pack(0,3) = fx(0);
            pack(1,3) = fx(1);
            pack(2,3) = fx(2);
            pose = globalTransforms[i] - pack; // Only minus t component from transform with rotated jVector
        }

        // Generate transform from weights
        for(int b=0; b<4; b++){
            BlockMatrix& block = blocks[b];
            for(int i=0; i<24; i++){
                block.col(i) = transforms[i].row(b);
            }
            //weightedBlockMatrix1[b] = Eigen::MatrixXf::Zero(4,mV.rows());
            weightedBlockMatrix1[b] = block*mWeightsT; // Column x VSize
        }

        for(int b=0; b<4; b++){
            Eigen::MatrixXf& block = weightedBlockMatrix1[b];
            for(int i=0; i<mV.rows(); i++){
                mVTemp(i,b) = mV.row(i) * block.col(i);
            }
        }

        return true;
    }
};

int main(int argc, char *argv[])
{
    SMPL smpl;
    smpl.loadModelFromJSONFile("/home/ryaadhav/smpl_cpp/model.json");

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    bool x = smpl.updateModel();
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<std::endl;

    op::WRender3D render;
    render.initializationOnThread();
    std::shared_ptr<op::WObject> wObject1 = std::make_shared<op::WObject>();
    //wObject1->loadOBJFile("/home/raaj/project/","hello_smpl.obj","");
    wObject1->loadEigenData(smpl.mV, smpl.mF);
    wObject1->print();
    render.addObject(wObject1);
    bool sw = true;
    while(1){

        static float currAng = 0.;
        if(currAng >= 45) sw = false;
        else if(currAng <= -45) sw = true;
        if(sw) currAng += 0.5;
        else currAng -= 0.5;
        cout << currAng << endl;
        smpl.mPose(1,0) = (M_PI/180. * currAng);
        smpl.mPose(1,1) = (M_PI/180. * currAng);

        //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        smpl.updateModel();
        wObject1->loadEigenData(smpl.mVTemp, smpl.mF);
        wObject1->rebuild();
        //std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() <<std::endl;

        render.workOnThread();
    }
}

