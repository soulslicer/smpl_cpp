#include <iostream>
using namespace std;

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
#include <stickman.h>
#include <pf.h>

class StickmanTracker : public ParticleFilter{
public:
    // Stickman
    StickMan meanStickman;
    const float PI = M_PI;
    const float PI2 = PI*2;
    std::vector<StickMan> stickmanFilters;
    const int totalParams = meanStickman.bLengths+(meanStickman.bodyParts*3)+3;

    StickmanTracker(int particleCount, Renderer::Params params) : ParticleFilter(particleCount, totalParams)
    {
        stickmanFilters.resize(particleCount);
        for(StickMan& stickman : stickmanFilters){
            stickman = meanStickman;
        }

        float minFloat = std::numeric_limits<float>::min();
        float maxFloat = std::numeric_limits<float>::max();

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

        // Range
        Eigen::MatrixXf rangeMatrix(totalParams,2);
        rangeMatrix <<  -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // BODY
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // CHIP
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // LLEG
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // RLEG
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // LKNEE
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // RKNEE
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // LFOOT
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // RFOOT
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // FNECK
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // NECK
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // NOSE
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // HEAD
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // LSHOULDER2
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // RSHOULDER2
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // LELBOW
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // RELBOW
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // LWRIST
                        -PI2,PI2,   -PI2,PI2,   -PI2,PI2, // RWRIST
                        0.1,1.2, // BODY HIP LENGTH
                        0.1,1.2, // HIP WIDTH
                        0.1,1.2  // LEG KNEE LENGTH

                        -0.75,0.75,  -2.5,0,   -2.5,0, // LELBOW
                        -0.75,0.75,  0,2.5,    -2.5,0, // REBLOW
                        -1.56,1.56, -1.56,1.56,-1.56,1.56, // LWRIST
                        -1.56,1.56, -1.56,1.56,-1.56,1.56, // RWRIST
                        -1.56,1.56, -1.56,1.56,-1.56,1.56, // LFINGERS
                        -1.56,1.56, -1.56,1.56,-1.56,1.56, // RFINGERS
                        -15, 7, //S0
                        -1, 1, //S1
                        -7, 7, //S2
                        -7, 7, //S3
                        -7, 7, //S4
                        -5, 5, //S5
                        -5, 5, //S6
                        -10, 10, //S7
                        -10, 10, //S8
                        -10, 10, //S9
                        -10, 10, //TX
                        -10, 10, //TY
                        -100, 100; //TZ
        //setRange(rangeMatrix);
    }
};


int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin, end;

    StickMan sm = StickMan();
    Eigen::MatrixXf mvTemp = sm.forward();
    Eigen::MatrixXf mF;

    // Render
    WRender3D render;
    render.initializationOnThread();
    end= std::chrono::steady_clock::now();
    std::cout << "Time difference Setup = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;
    std::shared_ptr<WObject> wObject1 = std::make_shared<WObject>();
    wObject1->loadEigenData(mvTemp, mF);
    wObject1->print();
    render.addObject(wObject1);
    wObject1->rebuild(WObject::RENDER_POINTS, 10);

    while(1){
        render.workOnThread();
        Eigen::MatrixXf mVTemp2 = sm.forward();
        wObject1->clearOBJFile(true);
        wObject1->loadEigenData(mVTemp2, mF);
        wObject1->loadKT(sm.kintree);
        wObject1->rebuild(WObject::RENDER_POINTS, 10);
    }

}
