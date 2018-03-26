#ifndef SMPL_HPP
#define SMPL_HPP

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/Eigen/Eigen>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/reader.h>
#include <tensor.h>
#include <fstream>

class SMPL{
public:
    Eigen::MatrixXf mV, mVTemp1, mVTemp2;
    Eigen::MatrixXf mF;
    Eigen::MatrixXf mPose;
    Eigen::MatrixXf mKintreeTable;
    Eigen::MatrixXf mJ, mJTemp1, mJTemp2;
    Eigen::MatrixXf mTrans;
    Eigen::MatrixXf mWeights;
    Eigen::MatrixXf mWeightsT;
    Eigen::MatrixXf vertSymIdxs;
    Eigen::MatrixXf mBetas;
    Eigen::SparseMatrix<float> mJR;
    typedef Eigen::Matrix<float, 4, 24> BlockMatrix;
    std::vector<Eigen::MatrixXf> weightedBlockMatrix1;
    std::vector<BlockMatrix> blocks;
    TensorD<3> mShapedDirsTensor;

    SMPL(){
        blocks.resize(4);
        weightedBlockMatrix1.resize(4);
        mBetas.resize(10,1);
        mBetas.setZero();
    }

//    // Copy constructor
//    SMPL(const SMPL &smpl){
//        mV = smpl.mV;
//        mVTemp1 = smpl.mVTemp1;
//        mVTemp2 = smpl.mVTemp2;
//        mF = smpl.mF;
//        mPose = smpl.mPose;
//        mKintreeTable = smpl.mKintreeTable;
//        mJ = smpl.mJ;
//        mJTemp1 = smpl.mJTemp1;
//        mJTemp2 = smpl.mJTemp2;
//        mTrans = smpl.mTrans;
//        mWeights = smpl.mWeights;
//        mWeightsT = smpl.mWeightsT;
//        vertSymIdxs = smpl.vertSymIdxs;
//        mBetas = smpl.mBetas;
//        mJR = smpl.mJR;
//        mShapedDirsTensor = smpl.mShapedDirsTensor;
//        for(Eigen::MatrixXf& w : weightedBlockMatrix1) w.resize(4,mV.rows());
//    }

//    SMPL clone(){
//        SMPL smpl;
//        smpl.mPose.resize(24,3);
//        smpl.mPose = mPose;
//        for(Eigen::MatrixXf& w : weightedBlockMatrix1) w.resize(4,mV.rows());
//        return smpl;
//    }

    enum Part
    {
        BODY,   // 0
        LLEG,   // 1
        RLEG,   // 2
        LTORSO,    // 3
        LKNEE,  // 4
        RKNEE,  // 5
        MTORSO,  // 6
        LFOOT,  // 7
        RFOOT,  // 8
        UTORSO, // 9
        LLFOOT, // 10
        RRFOOT, // 11
        NECK,   // 12
        LSHOULDER,  // 13
        RSHOULDER,  // 14
        HEAD,   // 15
        LSHOULDER2, // 16
        RSHOULDER2, // 17
        LELBOW, // 18
        RELBOW, // 19
        LWRIST, // 20
        RWRIST, // 21
        LFINGERS, // 22
        RFINGERS, // 23
        TRANS, // 24
    };

    enum Shape
    {
        S0,
        S1,
        S2,
        S3,
        S4,
        S5,
        S6,
        S7,
        S8,
        S9
    };

    void setPose(Part part, Eigen::Vector3f vec){
        if(part == Part::TRANS){
            mTrans(0,0) = vec(0);
            mTrans(1,0) = vec(1);
            mTrans(2,0) = vec(2);
            return;
        }
        int row = -1;
        if(part == Part::BODY) row = 0;
        if(part == Part::LLEG) row = 1;
        if(part == Part::RLEG) row = 2;
        if(part == Part::LTORSO) row = 3;
        if(part == Part::LKNEE) row = 4;
        if(part == Part::RKNEE) row = 5;
        if(part == Part::MTORSO) row = 6;
        if(part == Part::LFOOT) row = 7;
        if(part == Part::RFOOT) row = 8;
        if(part == Part::UTORSO) row = 9;
        if(part == Part::LLFOOT) row = 10;
        if(part == Part::RRFOOT) row = 11;
        if(part == Part::HEAD) row = 12;
        if(part == Part::LSHOULDER) row = 13;
        if(part == Part::RSHOULDER) row = 14;
        if(part == Part::NECK) row = 15;
        if(part == Part::LSHOULDER2) row = 16;
        if(part == Part::RSHOULDER2) row = 17;
        if(part == Part::LELBOW) row = 18;
        if(part == Part::RELBOW) row = 19;
        if(part == Part::LWRIST) row = 20;
        if(part == Part::RWRIST) row = 21;
        if(part == Part::LFINGERS) row = 22;
        if(part == Part::RFINGERS) row = 23;
        mPose.row(row) = vec;
    }

    void setShape(Shape shape, float val){
        int row = -1;
        if(shape == Shape::S0) row = 0;
        if(shape == Shape::S1) row = 1;
        if(shape == Shape::S2) row = 2;
        if(shape == Shape::S3) row = 3;
        if(shape == Shape::S4) row = 4;
        if(shape == Shape::S5) row = 5;
        if(shape == Shape::S6) row = 6;
        if(shape == Shape::S7) row = 7;
        if(shape == Shape::S8) row = 8;
        if(shape == Shape::S9) row = 9;
        mBetas(row) = val;
    }

    bool loadTensorFromJSON(const Json::Value& json, TensorD<3>& t, bool debug = false){
        int depth = json.size();
        int rows = json[0].size();
        int cols = json[0][0].size();
        if(debug){
            cout << "D: " << depth;
            cout << " R: " << rows;
            cout << " C: " << cols << endl;
        }

        t.resize({depth,rows,cols});

        for(int d=0; d<depth; d++){
            for(int r=0; r<rows; r++){
                for(int c=0; c<cols; c++){
                    t(d,r,c) = json[d][r][c].asFloat();
                }
            }
        }

        return true;
    }

    bool loadEigenVecFromJSON(const Json::Value& json, std::vector<Eigen::MatrixXf>& t, bool debug = false){
        int depth = json.size();
        int rows = json[0].size();
        int cols = json[0][0].size();
        if(debug){
            cout << "D: " << depth;
            cout << " R: " << rows;
            cout << " C: " << cols << endl;
        }

        t.resize(depth);
        for(Eigen::MatrixXf& m : t){
            m.resize(rows, cols);
        }

        for(int d=0; d<depth; d++){
            Eigen::MatrixXf& m = t[d];
            for(int r=0; r<rows; r++){
                for(int c=0; c<cols; c++){
                    t[d](r,c) = json[d][r][c].asFloat();
                }
            }
        }

        return true;
    }

    bool loadEigenFromJSON(const Json::Value& json, Eigen::MatrixXf& m, bool neg = false){
        // Set Shape
        int rows = json.size();
        if(!rows) { cerr << "Matrix Has no Rows" << endl; return false;}
        int cols = json[0].size();
        if(cols == 0){
            m.resize(rows, 1);
            for(int i=0; i<rows; i++){
                m(i,0) = json[i].asFloat();
            }
            return true;
        }
        if(rows == 0) rows = 1;
        m.resize(rows, cols);

        // Load Data
        if(rows > 1){
            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                    m(i,j) = json[i][j].asFloat();
                    if(neg) m(i,j)*=-1;
                }
            }
        }else{
            throw std::runtime_error("Something wrong");
        }

        return true;
    }

    bool loadSparseFromJSON(const Json::Value& json, Eigen::SparseMatrix<float>& m, int rows, int cols, bool debug = false){
        Eigen::MatrixXf intern;
        loadEigenFromJSON(json, intern);
        std::vector<Eigen::Triplet<float>> tripletList;
        tripletList.reserve(intern.rows());
        for(int r=1; r<intern.rows(); r++){
            tripletList.push_back(Eigen::Triplet<float>(intern(r,0),intern(r,1),intern(r,2)));
            //cout << intern(r,1) << endl;
        }
        m = Eigen::SparseMatrix<float>(rows,cols);
        m.setFromTriplets(tripletList.begin(), tripletList.end());
        return true;
    }

    bool loadPoseFromJSONFile(std::string filePath){
        ifstream in(filePath);
        Json::Value root;
        in >> root;
        if(!root.size()){
            cerr << "Failed to load pose file" << endl;
            return false;
        }

        loadEigenFromJSON(root["pose"], mPose);
        loadEigenFromJSON(root["betas"], mBetas);
        loadEigenFromJSON(root["trans"], mTrans);
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

        loadTensorFromJSON(root["shapedirs"], mShapedDirsTensor);

        loadEigenFromJSON(root["f"], mF);

        loadEigenFromJSON(root["kintree_table"], mKintreeTable);

        loadEigenFromJSON(root["J"], mJ);
        mJTemp1 = mJ;
        mJTemp2 = mJ;

        loadEigenFromJSON(root["trans"], mTrans);

        loadEigenFromJSON(root["v_posed"], mV);
        mV.conservativeResize(mV.rows(),mV.cols()+1);
        mV.col(mV.cols()-1) = Eigen::VectorXf::Ones(mV.rows());
        mVTemp1 = mV;
        mVTemp2 = mV;
        for(Eigen::MatrixXf& w : weightedBlockMatrix1) w.resize(4,mV.rows());

        loadEigenFromJSON(root["weights"], mWeights);
        mWeightsT = mWeights.transpose();

        loadEigenFromJSON(root["vert_sym_idxs"], vertSymIdxs);

        loadSparseFromJSON(root["J_regressor"], mJR, 24, mV.rows());

        return true;
    }

    Eigen::Matrix4f rod(const Eigen::VectorXf& v, const Eigen::VectorXf& t){
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

    bool updateModel(bool jointsOnly = false){

        // Create parent link table
        // {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
        std::map<int,int> parent;
        for(int i=1; i<mKintreeTable.cols(); i++){
            int key = mKintreeTable(1,i); int val = mKintreeTable(0,i);
            parent[key] = val;

            cout << "{" << key << "," << val << "},";
        }

        std::vector<Eigen::Matrix4f> globalTransforms(24);
        std::vector<Eigen::Matrix4f> transforms(24);

        // Shape
        TensorD<3> AB = mShapedDirsTensor.dot(mBetas);
        for(int i=0; i<mVTemp1.rows(); i++){
            mVTemp1(i,0) = mV(i,0) + AB(i,0,0);
            mVTemp1(i,1) = mV(i,1) + AB(i,1,0);
            mVTemp1(i,2) = mV(i,2) + AB(i,2,0);
            mVTemp1(i,3) = 1;
        }

        // Shape J
        mJTemp2.row(0) = mJ.row(0);
        mJTemp1 = mJR * mVTemp1;
        //mJTemp1 = mJ;

        cout << mJTemp1 << endl;

        // Body pose
        Eigen::Matrix4f& bodyPose = globalTransforms[0];
        bodyPose = rod(mPose.row(0), mJTemp1.row(0));

        // Global Transforms
        for(int i=1; i<globalTransforms.size(); i++){
            Eigen::Matrix4f& pose = globalTransforms[i];
            pose = globalTransforms[parent[i]] * rod(mPose.row(i), mJTemp1.row(i) - mJTemp1.row(parent[i]));
            mJTemp2(i,0) = pose(0,3);
            mJTemp2(i,1) = pose(1,3);
            mJTemp2(i,2) = pose(2,3);
        }

        // Trans
        for(int i=0; i<mJTemp2.rows(); i++){
            mJTemp2(i,0) = mJTemp2(i,0) + mTrans(0,0);
            mJTemp2(i,1) = mJTemp2(i,1) + mTrans(1,0);
            mJTemp2(i,2) = mJTemp2(i,2) + mTrans(2,0);
        }
        if(jointsOnly) return true;

        // Transforms
        for(int i=0; i<transforms.size(); i++){
            Eigen::Matrix4f& pose = transforms[i];
            Eigen::Vector4f jZero;
            jZero << mJTemp1(i,0), mJTemp1(i,1), mJTemp1(i,2), 0;
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
            weightedBlockMatrix1[b] = block*mWeightsT; // Column x VSize ~2ms
        }

        // Transform vertices with weight matrix
        for(int b=0; b<4; b++){
            Eigen::MatrixXf& block = weightedBlockMatrix1[b];
            for(int i=0; i<mV.rows(); i++){
                mVTemp2(i,b) = mVTemp1.row(i) * block.col(i);
            }
        }

        // Final transform
        for(int i=0; i<mVTemp2.rows(); i++){
            mVTemp2(i,0) = mVTemp2(i,0) + mTrans(0,0);
            mVTemp2(i,1) = mVTemp2(i,1) + mTrans(1,0);
            mVTemp2(i,2) = mVTemp2(i,2) + mTrans(2,0);
        }

        return true;
    }
};

#endif
