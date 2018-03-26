#include <iostream>
using namespace std;

// python code/fit_3d.py $PWD --viz
// cd code;
// python smpl_webuser/hello_world/render_smpl.py

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <renderer.hpp>
#include <eigen3/Eigen/Eigen>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/reader.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iomanip>
#include <chrono>

#include <trackbar.h>
#include <tensor.h>
#include <smpl.h>

#include <GLFW/glfw3.h>

#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

#include <op.h>

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

float d2r(float deg){
    return (M_PI/180.)*deg;
}

class StickMan{
public:

    const int BODY = 0;   // 0
    const int CHIP = 1;
    const int LLEG = 2;  // 1
    const int RLEG = 3;   // 2
    const int LKNEE = 4;  // 4
    const int RKNEE = 5;  // 5
    const int LFOOT = 6;  // 7
    const int RFOOT = 7;  // 8
    const int FNECK = 8; // 9
    const int NECK = 9;   // 12
    const int  HEAD = 10;   // 15
    const int LSHOULDER2 = 11; // 16
    const int RSHOULDER2 = 12; // 17
    const int LELBOW = 13; // 18
    const int RELBOW = 14; // 19
    const int LWRIST = 15; // 20
    const int RWRIST = 16; // 21


    int bodyParts = 17;
    Eigen::MatrixXf theta = Eigen::MatrixXf::Zero(bodyParts,3);
    Eigen::MatrixXf beta = Eigen::MatrixXf::Zero(10,1);
    std::map<int,int> kintree;
    StickMan(){
        beta << 0.3,0.05,0.3,0.3,0.2,0.2,0.2,0.2,1,1;
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
                   {HEAD,NECK},
                   {LELBOW,LSHOULDER2},
                   {RELBOW,RSHOULDER2},
                   {LWRIST,LELBOW},
                   {RWRIST,RELBOW}};
    }

    Eigen::MatrixXf forward(){
        Eigen::MatrixXf mJ = Eigen::MatrixXf::Zero(bodyParts,3);
        Eigen::MatrixXf mJ2 = Eigen::MatrixXf::Zero(bodyParts,3);


        mJ <<   0,0,0,                              //    BODY,   // 0
                0, -0.327, 0.0, // CHIP 1
                0.0564113,   -0.323066,   0.0109195, //    LLEG,   // 2
                -0.0624814,   -0.331303,   0.0150424, //    RLEG,   // 3
                0.099859,   -0.709545,   0.0189577, //    LKNEE,  // 4
                -0.105741,   -0.714992,   0.0101987, //    RKNEE,  // 5
                0.0850685,     -1.1364,  -0.0184713, //    LFOOT,  // 6
                -0.0866823,    -1.13502,  -0.0243684, //    RFOOT,  // 7
                0.00449099,    0.197604,    0.019872, //    FNECK, // 8

                -0.00890061,    0.289232,  -0.0135916, //    NECK,   // 12

                0.00121335,    0.378176,   0.0368132,//    HEAD,   // 15
                0.199114,     0.23681,  -0.0180674, //    LSHOULDER2, // 16
                -0.191692,    0.236926,   -0.012305, //    RSHOULDER2, // 17
                0.454449,     0.22116,  -0.0410182,//    LELBOW, // 18
                -0.451818,    0.222556,  -0.0435726, //    RELBOW, // 19
                0.720163,    0.233858,  -0.0483931, //    LWRIST, // 20
                -0.72092,    0.229351,  -0.0496015; //    RWRIST, // 21


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

        //        Eigen::MatrixXf mJ = Eigen::MatrixXf::Zero(bodyParts,3);
        //        mJ.row(0) = Eigen::Vector3f(0,0,0);

        //        for (auto& kv : kintree) {
        //            int parent = kv.second; int child = kv.first;
        //            // Root to neck
        //            if(parent == 0 && child == 1){
        //                mJ.row(child) = Eigen::Vector3f(mJ(parent,0) + beta(0),mJ(parent,1),mJ(parent,2));
        //            }
        //            // Neck to lshoulder
        //            if(parent == 1 && child == 2){
        //                mJ.row(child) = Eigen::Vector3f(mJ(parent,0) + beta(1),mJ(parent,1),mJ(parent,2));
        //            }
        //            // lshoulder to lelbow
        //            if(parent == 2 && child == 3){
        //                mJ.row(child) = Eigen::Vector3f(mJ(parent,0) + beta(2),mJ(parent,1),mJ(parent,2));
        //            }
        //            // lshoulder to lwrist
        //            if(parent == 3 && child == 4){
        //                mJ.row(child) = Eigen::Vector3f(mJ(parent,0) + beta(3),mJ(parent,1),mJ(parent,2));
        //            }

        ////            // lshoulder to lwrist
        ////            if(parent == 1 && child == 5){
        ////                mJ.row(child) = Eigen::Vector3f(mJ(parent,0) - beta(1),mJ(parent,1),mJ(parent,2));
        ////            }
        //        }

        //        // Compute new mJ from mJ
        //        Eigen::MatrixXf mJ2 = mJ;
        //        std::vector<Eigen::Matrix4f> globalTransforms(bodyParts);
        //        Eigen::Matrix4f& rootPose = globalTransforms[0];
        //        rootPose = rod(theta.row(0), mJ.row(0));

        //        // Global Transforms
        //        for(int i=1; i<globalTransforms.size(); i++){
        //            Eigen::Matrix4f& pose = globalTransforms[i];
        //            pose = globalTransforms[kintree[i]] * rod(theta.row(i), mJ.row(i) - mJ.row(kintree[i]));
        //            mJ2(i,0) = pose(0,3);
        //            mJ2(i,1) = pose(1,3);
        //            mJ2(i,2) = pose(2,3);

        //                cout << i << "->" << kintree[i] << endl;
        //                cout << theta.row(i) << endl;
        //                cout << pose << endl;

        //        }

        //return mJ2;
    }
};

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin, end;

    StickMan sm = StickMan();
    Eigen::MatrixXf mvTemp = sm.forward();
    Eigen::MatrixXf mF;

    // Trackbar
    DoubleTrackManager trackManager;
    trackManager.setupWindow("Track");
    trackManager.addTrack("T0",M_PI*4);
    trackManager.addTrack("T1",M_PI*4);
    trackManager.addTrack("T2",M_PI*4);
    trackManager.addTrack("T3",M_PI*4);
    trackManager.addTrack("T4",M_PI*4);
    trackManager.addTrack("T5",M_PI*4);

    trackManager.addTrack("T11",M_PI*4);
    trackManager.addTrack("T12",M_PI*4);
    trackManager.addTrack("T13",M_PI*4);
    trackManager.addTrack("T14",M_PI*4);

    trackManager.addTrack("B0",M_PI*4, 0.3);
    trackManager.addTrack("B1",M_PI*4, 0.1);
    trackManager.addTrack("B2",M_PI*4, 0.3);
    trackManager.addTrack("B3",M_PI*4, 0.3);
    trackManager.addTrack("B4",M_PI*4, 0.2);
    trackManager.addTrack("B5",M_PI*4, 0.2);

    begin = std::chrono::steady_clock::now();
    WRender3D render;
    render.initializationOnThread();
    end= std::chrono::steady_clock::now();
    std::cout << "Time difference Setup = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;
    std::shared_ptr<WObject> wObject1 = std::make_shared<WObject>();
    //wObject1->loadOBJFile("/home/raaj/project/","hello_smpl.obj","");
    wObject1->loadEigenData(mvTemp, mF);
    wObject1->print();
    render.addObject(wObject1);
    wObject1->rebuild(WObject::RENDER_POINTS, 10);
    //wObject1->rebuildVArr(op::WObject::RENDER_NORMAL);

    while(1){

        render.workOnThread();

        trackManager.spin();
        if(trackManager.changeOccured()){
            sm.theta.row(0) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T0"));
            sm.theta.row(1) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T1"));
            sm.theta.row(2) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T2"));
            sm.theta.row(3) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T3"));
            sm.theta.row(4) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T4"));
            sm.theta.row(5) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T5"));
            sm.theta.row(11) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T11"));
            sm.theta.row(12) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T12"));
            sm.theta.row(13) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T13"));
            sm.theta.row(14) = Eigen::Vector3f(0,0,trackManager.getTrackValue("T14"));

            sm.beta(0,0) = trackManager.getTrackValue("B0");
            sm.beta(1,0) = trackManager.getTrackValue("B1");
            sm.beta(2,0) = trackManager.getTrackValue("B2");

            sm.beta(3,0) = trackManager.getTrackValue("B3");
            sm.beta(4,0) = trackManager.getTrackValue("B4");
            sm.beta(5,0) = trackManager.getTrackValue("B5");
            //sm.beta(0,0) = trackManager.getTrackValue("X");

            Eigen::MatrixXf mVTemp2 = sm.forward();
            wObject1->clearOBJFile(true);
            wObject1->loadEigenData(mVTemp2, mF);
            wObject1->loadKT(sm.kintree);
            wObject1->rebuild(WObject::RENDER_POINTS, 10);
            //cout << "[ " << trackManager.getTrackValue("LLEG_X") << " " << trackManager.getTrackValue("LLEG_Y") << " " << trackManager.getTrackValue("LLEG_Z") << " ]" << endl;
            //smpl.setPose(SMPL::BODY, Eigen::Vector3f(trackManager.getTrackValue("LLEG_X"), trackManager.getTrackValue("LLEG_Y"), trackManager.getTrackValue("LLEG_Z")));
            //smpl.setShape(SMPL::S0,trackManager.getTrackValue("LLEG_X"));
            //smpl.updateModel();
            //wObject1->loadEigenData(smpl.mVTemp2, smpl.mF);
            //wObject1->rebuild(WObject::RENDER_NORMAL);
        }


    }
}
