#ifndef STICKMAN_HPP
#define STICKMAN_HPP

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/Eigen/Eigen>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/reader.h>
#include <tensor.h>
#include <fstream>

class StickMan{
public:

    int BODY = 0;   // 0
    int CHIP = 1;
    int LLEG = 2;  // 1
    int RLEG = 3;   // 2
    int LKNEE = 4;  // 4
    int RKNEE = 5;  // 5
    int LFOOT = 6;  // 7
    int RFOOT = 7;  // 8
    int FNECK = 8; // 9
    int NECK = 9;   // 12
    int NOSE = 10;
    int HEAD = 11;   // 15
    int LSHOULDER2 = 12; // 16
    int RSHOULDER2 = 13; // 17
    int LELBOW = 14; // 18
    int RELBOW = 15; // 19
    int LWRIST = 16; // 20
    int RWRIST = 17; // 21

    int bodyParts = 18;
    int bLengths = 10;

    Eigen::MatrixXf theta = Eigen::MatrixXf::Zero(bodyParts,3);
    Eigen::MatrixXf beta = Eigen::MatrixXf::Zero(10,1);
    std::map<int,int> kintree;
    StickMan(){
        beta << 0.3,0.05,0.3,0.3,0.2,0.2,0.2,0.2,0.1,0.1;
        kintree = {{CHIP, BODY},
                   {LLEG,CHIP},
                   {RLEG,CHIP},
                   {LKNEE,LLEG},
                   {RKNEE,RLEG},
                   {LFOOT,LKNEE},
                   {RFOOT,RKNEE},
                   {FNECK,BODY},
                   {NECK,FNECK},
                   {LSHOULDER2,FNECK},
                   {RSHOULDER2,FNECK},
                   {LELBOW,LSHOULDER2},
                   {RELBOW,RSHOULDER2},
                   {LWRIST,LELBOW},
                   {RWRIST,RELBOW},
                   {NOSE,NECK},
                   {HEAD,NOSE},
                  };
    }

    Eigen::MatrixXf forward(){
        Eigen::MatrixXf mJ = Eigen::MatrixXf::Zero(bodyParts,3);
        Eigen::MatrixXf mJ2 = Eigen::MatrixXf::Zero(bodyParts,3);

        for(int i=1; i<bodyParts; i++){
            Eigen::Vector3f p = mJ.row(kintree[i]);
            if(i==CHIP) mJ.row(i) = p + Eigen::Vector3f(0,-1,0)*beta(0);
            if(i==LLEG) mJ.row(i) = p + Eigen::Vector3f(1,0,0)*beta(1);
            if(i==RLEG) mJ.row(i) = p + Eigen::Vector3f(-1,0,0)*beta(1);
            if(i==LKNEE || i==RKNEE) mJ.row(i) = p + Eigen::Vector3f(0,-1,0)*beta(2);
            if(i==LFOOT || i==RFOOT) mJ.row(i) = p + Eigen::Vector3f(0,-1,0)*beta(3);
            if(i==FNECK) mJ.row(i) = p + Eigen::Vector3f(0,1,0)*beta(4);
            if(i==LSHOULDER2) mJ.row(i) = p + Eigen::Vector3f(1,0,0)*beta(5);
            if(i==RSHOULDER2) mJ.row(i) = p + Eigen::Vector3f(-1,0,0)*beta(5);
            if(i==LELBOW) mJ.row(i) = p + Eigen::Vector3f(1,0,0)*beta(6);
            if(i==RELBOW) mJ.row(i) = p + Eigen::Vector3f(-1,0,0)*beta(6);
            if(i==LWRIST) mJ.row(i) = p + Eigen::Vector3f(1,0,0)*beta(7);
            if(i==RWRIST) mJ.row(i) = p + Eigen::Vector3f(-1,0,0)*beta(7);
            if(i==NECK) mJ.row(i) = p + Eigen::Vector3f(0,1,0)*beta(8);
            if(i==NOSE) mJ.row(i) = p + Eigen::Vector3f(0,1,0)*beta(9);
            if(i==HEAD) mJ.row(i) = p + Eigen::Vector3f(0,1,0)*beta(9);
        }

        // Compute new mJ from mJ
        mJ2.row(0) = mJ.row(0);
        std::vector<Eigen::Matrix4f> globalTransforms(bodyParts);
        Eigen::Matrix4f& rootPose = globalTransforms[0];
        rootPose = rod(theta.row(0), mJ.row(0));

        // Global Transforms
        for(int i=1; i<globalTransforms.size(); i++){
            Eigen::Matrix4f& pose = globalTransforms[i];
            pose = globalTransforms[kintree[i]] * rod(theta.row(i), mJ.row(i) - mJ.row(kintree[i]));
            mJ2(i,0) = pose(0,3);
            mJ2(i,1) = pose(1,3);
            mJ2(i,2) = pose(2,3);
        }

        return mJ2;
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
};

#endif
