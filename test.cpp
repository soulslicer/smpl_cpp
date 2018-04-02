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

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin, end;

//    cv::Mat im1 = cv::imread("/home/ryaadhav/smpl_cpp/data/00001_image.png");
//    OpenPose op;
//    op.forward(im1);
//    begin = std::chrono::steady_clock::now();
//    op.forward(im1);
//    end= std::chrono::steady_clock::now();
//    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;
//    op.forward(im1, true);
//    //exit(-1);

    bool active = false;
    bool track = true;
    bool joints = false;

    SMPL smpl;
    smpl.loadModelFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/male_model.json");
    //std::cout.setstate(std::ios_base::failbit);
    smpl.updateModel(joints);
    //std::cout.clear();

    //smpl.loadPoseFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/00001_body.json");
    //smpl.setShape(SMPL::S3, 20);
    //smpl.mJ(17,2) = 5;
    //smpl.setPose(SMPL::RSHOULDER2, Eigen::Vector3f(M_PI/180. * 20, M_PI/180. * 20, M_PI/180. * 20));
    //smpl.setPose(SMPL::LLEG, Eigen::Vector3f(M_PI/180. * 20, M_PI/180. * 20, M_PI/180. * 20));

    begin = std::chrono::steady_clock::now();
    smpl.updateModel(joints);
    end= std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;

    // Trackbar
    DoubleTrackManager trackManager;
    if(track){
        trackManager.setupWindow("Track");
        trackManager.addTrack("LLEG_X",M_PI*4);
        trackManager.addTrack("LLEG_Y",M_PI*4);
        trackManager.addTrack("LLEG_Z",M_PI*4);
    }

    begin = std::chrono::steady_clock::now();
    WRender3D render;
    render.initializationOnThread();
    end= std::chrono::steady_clock::now();
    std::cout << "Time difference Setup = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;
    std::shared_ptr<WObject> wObject1 = std::make_shared<WObject>();
    //wObject1->loadOBJFile("/home/raaj/project/","hello_smpl.obj","");

    wObject1->loadEigenData(smpl.mVTemp2, smpl.mF);
    wObject1->print();
    render.addObject(wObject1);
    if(!joints)
    wObject1->rebuild(WObject::RENDER_POINTS);
    //wObject1->rebuildVArr(op::WObject::RENDER_NORMAL);

    std::shared_ptr<WObject> wObject2 = std::make_shared<WObject>();
    if(joints){
        Eigen::MatrixXf empt;
        wObject2->loadEigenData(smpl.mJTemp2, empt);
        render.addObject(wObject2);
        wObject2->rebuild(WObject::RENDER_POINTS, 10);
    }

    bool sw = true;
    while(1){

        if(active){
            static float currAng = 0.;
            static float currB = 0.;
            if(currAng >= 45) sw = false;
            else if(currAng <= -45) sw = true;
            if(sw) {
                currAng += 0.5;
                currB += 0.1;
            }
            else {
                currAng -= 0.5;
                currB -= 0.1;
            }
            smpl.setPose(SMPL::LSHOULDER, Eigen::Vector3f(M_PI/180. * currAng, M_PI/180. * currAng, 0));
            smpl.setPose(SMPL::RSHOULDER, Eigen::Vector3f(M_PI/180. * currAng, M_PI/180. * currAng, 0));
            smpl.setPose(SMPL::NECK, Eigen::Vector3f(0, 0, M_PI/180. * currAng));
            smpl.setShape(SMPL::S3, currB);

            begin = std::chrono::steady_clock::now();
            smpl.updateModel(joints);
            wObject1->loadEigenData(smpl.mVTemp2, smpl.mF);
            if(!joints)
            wObject1->rebuildVArr(WObject::RENDER_NORMAL);
            end= std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;

            if(joints){
                Eigen::MatrixXf empt;
                wObject2->loadEigenData(smpl.mJTemp2, empt);
                wObject2->rebuild(WObject::RENDER_POINTS, 10);
            }
        }

        render.workOnThread();

        if(track){
            trackManager.spin();
            if(trackManager.changeOccured()){
                cout << "[ " << trackManager.getTrackValue("LLEG_X") << " " << trackManager.getTrackValue("LLEG_Y") << " " << trackManager.getTrackValue("LLEG_Z") << " ]" << endl;
                smpl.setPose(SMPL::LLFOOT, Eigen::Vector3f(trackManager.getTrackValue("LLEG_X"), trackManager.getTrackValue("LLEG_Y"), trackManager.getTrackValue("LLEG_Z")));
                //smpl.setShape(SMPL::S0,trackManager.getTrackValue("LLEG_X"));
                smpl.updateModel(joints);
                wObject1->loadEigenData(smpl.mVTemp2, smpl.mF);
                if(!joints)
                wObject1->rebuild(WObject::RENDER_NORMAL);

                if(joints){
                    Eigen::MatrixXf empt;
                    wObject2->loadEigenData(smpl.mJTemp2, empt);
                    wObject2->rebuild(WObject::RENDER_POINTS, 10);
                }
            }
        }

    }
}
