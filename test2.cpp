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

const double singularity_cutoff = M_PI/2 * 0.985;

Eigen::Vector3f quat_to_euler(Eigen::Quaternionf q)
{
    Eigen::Vector3f euler;

    // [q0 q1 q2 q3] is in w, x, y, z order
    const double q0 = q.w();
    const double q1 = q.x();
    const double q2 = q.y();
    const double q3 = q.z();

    euler[0] = atan2(2*(q0*q1 + q2*q3), 1-2*(q1*q1 + q2*q2));
    euler[1] = asin(2*(q0*q2 - q3*q1));
    euler[2] = atan2(2*(q0*q3 + q1*q2), 1-2*(q2*q2 + q3*q3));

    // Tentatively handle singularities.
    if (euler[1] > singularity_cutoff || euler[1] < -singularity_cutoff) {
        euler[0] = atan2(q3, q0);
        euler[2] = 0;
    }

    return euler;
}

// body 3-2-1
Eigen::Quaternionf euler_to_quat(float r, float p, float y)
{
    return Eigen::Quaternionf(Eigen::AngleAxisf(y, Eigen::Vector3f::UnitZ())
                            * Eigen::AngleAxisf(p, Eigen::Vector3f::UnitY())
                            * Eigen::AngleAxisf(r, Eigen::Vector3f::UnitX()));
}

Eigen::AngleAxisf euler_to_aa(float r, float p, float y)
{
    return Eigen::AngleAxisf(euler_to_quat(r,p,y));
}

void swing_twist(const Eigen::Quaternionf& q, const Eigen::Vector3f& vt,
                 Eigen::Quaternionf& swing, Eigen::Quaternionf& twist) {
    Eigen::Vector3f p = vt * (q.x() * vt[0] + q.y() * vt[1] + q.z() * vt[2]);
    twist = Eigen::Quaternionf(q.w(), p[0], p[1], p[2]);
    twist.normalize();
    swing = q * twist.conjugate();

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
    const int NOSE = 10;
    const int HEAD = 11;   // 15
    const int LSHOULDER2 = 12; // 16
    const int RSHOULDER2 = 13; // 17
    const int LELBOW = 14; // 18
    const int RELBOW = 15; // 19
    const int LWRIST = 16; // 20
    const int RWRIST = 17; // 21

    int bodyParts = 18;
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
    trackManager.addTrack("BODY0",M_PI*4);

    trackManager.addTrack("CHIPX",M_PI*4);
    trackManager.addTrack("CHIPY",M_PI*4);
    trackManager.addTrack("CHIPZ",M_PI*4);

//    trackManager.addTrack("LLEG2",M_PI*4);
//    trackManager.addTrack("RLEG3",M_PI*4);
//    trackManager.addTrack("LKNEE4",M_PI*4);
//    trackManager.addTrack("RKNEE5",M_PI*4);
//    trackManager.addTrack("FNECK8",M_PI*4);
//    trackManager.addTrack("NECK9",M_PI*4);
//    trackManager.addTrack("NOSE10",M_PI*4);
//    trackManager.addTrack("HEAD11",M_PI*4);
//    trackManager.addTrack("LSHOULDER12",M_PI*4);
//    trackManager.addTrack("RSHOULDER13",M_PI*4);
//    trackManager.addTrack("LELBOW14",M_PI*4);
//    trackManager.addTrack("RELBOW15",M_PI*4);

//    trackManager.addTrack("B0",M_PI*4, 0.3);
//    trackManager.addTrack("B1",M_PI*4, 0.1);
//    trackManager.addTrack("B2",M_PI*4, 0.3);
//    trackManager.addTrack("B3",M_PI*4, 0.3);
//    trackManager.addTrack("B4",M_PI*4, 0.2);
//    trackManager.addTrack("B5",M_PI*4, 0.2);

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

            sm.theta.row(0) = Eigen::Vector3f(0,0,trackManager.getTrackValue("BODY0"));

            // 1 - CHIP [-120, -45, -45] [45, 45, 45]
            // 2 - LLEG [-120 -45 -40] [30 45 40]
            // 3 - RLEG [-120 -45 -40] [30 45 40]
            // 4 - LKNEE [0, -20, -20] [130, 20, 20]
            // 5 - LKNEE [0, -20, -20] [130, 20, 20]

            int r = 5;
            sm.theta.row(r) = Eigen::Vector3f(trackManager.getTrackValue("CHIPX"),trackManager.getTrackValue("CHIPY"),trackManager.getTrackValue("CHIPZ"));
            cout << sm.theta.row(r) * 180/M_PI << endl;

//            Eigen::AngleAxisf aa = euler_to_aa(sm.theta(r,0),sm.theta(r,1),sm.theta(r,2));
//            cout << aa.axis().transpose() << " " << aa.angle() << endl;

//            sm.theta.row(1) = Eigen::Vector3f(0,0,trackManager.getTrackValue("CHIP1"));
//            sm.theta.row(2) = Eigen::Vector3f(0,0,trackManager.getTrackValue("LLEG2"));
//            sm.theta.row(3) = Eigen::Vector3f(0,0,trackManager.getTrackValue("RLEG3"));
//            sm.theta.row(4) = Eigen::Vector3f(0,0,trackManager.getTrackValue("LKNEE4"));
//            sm.theta.row(5) = Eigen::Vector3f(0,0,trackManager.getTrackValue("RKNEE5"));

//            sm.theta.row(8) = Eigen::Vector3f(0,0,trackManager.getTrackValue("FNECK8"));
//            sm.theta.row(9) = Eigen::Vector3f(0,0,trackManager.getTrackValue("NECK9"));
//            sm.theta.row(10) = Eigen::Vector3f(0,0,trackManager.getTrackValue("NOSE10"));

//            sm.theta.row(11) = Eigen::Vector3f(0,0,trackManager.getTrackValue("HEAD11"));

//            sm.theta.row(12) = Eigen::Vector3f(0,0,trackManager.getTrackValue("LSHOULDER12"));
//            sm.theta.row(13) = Eigen::Vector3f(0,0,trackManager.getTrackValue("RSHOULDER13"));
//            sm.theta.row(14) = Eigen::Vector3f(0,0,trackManager.getTrackValue("LELBOW14"));
//            sm.theta.row(15) = Eigen::Vector3f(0,0,trackManager.getTrackValue("RELBOW15"));

//            sm.beta(0,0) = trackManager.getTrackValue("B0");
//            sm.beta(1,0) = trackManager.getTrackValue("B1");
//            sm.beta(2,0) = trackManager.getTrackValue("B2");

//            sm.beta(3,0) = trackManager.getTrackValue("B3");
//            sm.beta(4,0) = trackManager.getTrackValue("B4");
//            sm.beta(5,0) = trackManager.getTrackValue("B5");
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
