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

#include <pf.h>

void overlayImage(cv::Mat* src, cv::Mat* overlay, const cv::Point& location)
{
    for (int y = max(location.y, 0); y < src->rows; ++y)
    {
        int fY = y - location.y;

        if (fY >= overlay->rows)
            break;

        for (int x = max(location.x, 0); x < src->cols; ++x)
        {
            int fX = x - location.x;

            if (fX >= overlay->cols)
                break;

            double opacity = ((double)overlay->data[fY * overlay->step + fX * overlay->channels() + 3]) / 255;

            for (int c = 0; opacity > 0 && c < src->channels(); ++c)
            {
                unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
                unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
                src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
            }
        }
    }
}

class SMPLTracker : public ParticleFilter{
public:
    SMPL meanSMPL;
    cv::Mat input;
    const float PI = M_PI;
    const float PI2 = PI*2;
    std::vector<SMPL> smplFilters;

    SMPLTracker(int particleCount, int paramsCount) : ParticleFilter(particleCount, paramsCount)
    {
        meanSMPL.loadModelFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/male_model.json");
        meanSMPL.updateModel();

        smplFilters.resize(particleCount);
        for(SMPL& smpl : smplFilters){
            smpl = meanSMPL;
        }

        float minFloat = std::numeric_limits<float>::min();
        float maxFloat = std::numeric_limits<float>::max();

        // Range
        Eigen::MatrixXf rangeMatrix(85,2);
        rangeMatrix <<  -maxFloat,maxFloat,   -maxFloat,maxFloat,   -maxFloat,maxFloat, // BODY
                -1.2,0.5,   -1,1,       -1,1,       // LLEG
                -1.2,0.5,   -1,1,       -1,1,       // RLEG
                -0.5,1.5,   -0.5,0.5,   -0.7,0.7,   // LTORSO
                0,1.56,     0,0,        0,0,        // LKNEE
                0,1.56,     0,0,        0,0,        // RKNEE
                -0.6,0.6,   -0.6,0.6,   -0.6,0.6, // MTORSO
                -0.7,0.7,   -0.7,0.7,   -0.7,0.7,        // LFOOT
                -0.7,0.7,   -0.7,0.7,   -0.7,0.7,        // RFOOT
                -0.6,0.6,   -0.6,0.6,   -0.6,0.6, // UTORSO
                -0.6,0.6,   -0.6,0.6,   -0.6,0.6, // LLFOOT
                -0.6,0.6,   -0.6,0.6,   -0.6,0.6, // RRFOOT
                -1,1,       -1,1,       -1,1, // HEAD
                -0.5,0.5,   -1.8,0.1,   -1.57,1.57, // LSHOULDER
                -0.5,0.5,   -0.1,1.8,   -1.57,1.57, // RSHOULDER
                -0.7,0.7,   -1.57,1.57, -0.7,0.7, // NECK
                -0.5,0.5,   -1.8,0.1,   -1.57,1.57, // LSHOULDER2
                -0.5,0.5,   -0.1,1.8,   -1.57,1.57, // RSHOULDER2
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
        setRange(rangeMatrix);
    }

    void setSMPLFromPF(SMPL& smpl, Eigen::MatrixXf mean, bool jointsOnly = false){
        for(int i=0; i<24; i++){
            smpl.mPose.row(i) = Eigen::Vector3f(mean(i*3 + 0, 0), mean(i*3 + 1, 0), mean(i*3 + 2, 0));
        }
        for(int i=0; i<10; i++){
            smpl.mBetas(i,0) = mean(72 + i, 0);
        }
        for(int i=0; i<3; i++){
            smpl.mTrans(i,0) = mean(82 + i, 0);
        }
        smpl.updateModel(jointsOnly);
    }

    cv::Point2i project(Eigen::Vector3f point, Renderer::Params& params){
        cv::Point2i pixel;
        pixel.x =(int)(((params.fl*point(0))/point(2)) + params.cx);
        pixel.y =(int)(((params.fl*point(1))/point(2)) + params.cy);
        return pixel;
    }

    float l2distance(const cv::Point2i& a, const cv::Point2i& b){
        return sqrt(pow(a.x-b.x,2) + pow(a.y-b.y,2));
    }

    cv::Mat weightFunction(Eigen::MatrixXf opOutput, Renderer::Params& params){
        cv::Mat debugImg = input.clone();

        ParticleFilter::Probability pixelReprojProb(5);
        std::vector<SMPL>& smplFiltersSC = smplFilters;
        Eigen::MatrixXf weights = Eigen::MatrixXf::Zero(weightVector_.rows(),weightVector_.cols());
#pragma omp parallel for shared(smplFiltersSC, weights)
        for(int i=0; i<particleCount_; i++){
            SMPL& smpl = smplFiltersSC[i];
            setSMPLFromPF(smpl, stateMatrix_.col(i), true);

            for(int j=0; j<24; j++){
                Eigen::Vector3f hypoPoint(smpl.mJTemp2(j,0),smpl.mJTemp2(j,1),smpl.mJTemp2(j,2));
                cv::Point hypoPix = project(hypoPoint, params);
                cv::circle(debugImg, hypoPix, 3, cv::Scalar(0,255,0), CV_FILLED);

                int opIndex = -1;
                if(j == 1) opIndex = 11;
                if(j == 2) opIndex = 8;
                if(j == 4) opIndex = 12;
                if(j == 5) opIndex = 9;
                if(j == 7) opIndex = 13;
                if(j == 8) opIndex = 10;
                if(j == 15) opIndex = 1;
                if(j == 16) opIndex = 5;
                if(j == 17) opIndex = 2;
                if(j == 18) opIndex = 6;
                if(j == 19) opIndex = 3;
                if(j == 20) opIndex = 7;
                if(j == 21) opIndex = 4;
                if(opIndex < 0) continue;

                cv::Point truthPix(opOutput(opIndex,0),opOutput(opIndex,1));
                cv::circle(debugImg, truthPix, 3, cv::Scalar(255,0,0), CV_FILLED);
                float reprojErr = l2distance(truthPix, hypoPix);
                weights(i,0) += pixelReprojProb.getProbability(reprojErr).log;
            }
        }
        weightVector_ = weights;

        return debugImg;
    }

    cv::Mat draw(){
        cv::Mat img = input.clone();
        for(int i=0; i<particleCount_; i++){
            cv::Point p(stateMatrix_(0,i),stateMatrix_(1,i));
            cv::circle(img, p, 1, cv::Scalar(255,0,0),CV_FILLED);
        }

        Eigen::MatrixXf mean = computeMean();
        cv::Point mP(mean(0,0),mean(1,0));
        cv::circle(img, mP, 5, cv::Scalar(255,255,0),CV_FILLED);
        return img;
    }

    void computeMeanSMPL(){
        Eigen::MatrixXf mean = this->computeMean();
        setSMPLFromPF(meanSMPL, mean);
    }

};

Eigen::MatrixXf convertOPtoEigen(op::Array<float>& opOutput){
    Eigen::MatrixXf eigen(opOutput.getSize()[1],opOutput.getSize()[2]);
    for(int r=0; r<eigen.rows(); r++){
        for(int c=0; c<eigen.cols(); c++){
            eigen(r,c) = opOutput[eigen.cols()*r + c];
        }
    }
    return eigen;
}

void testPF2(){
    std::chrono::steady_clock::time_point begin, end;
    glfwInit();
    OpenPose op;
    SMPLTracker pf(300,85);
    Renderer renderer;
    Renderer::Params params;

    bool once = false;
    cv::VideoCapture capture("/home/ryaadhav/Desktop/video.mp4");
    if( !capture.isOpened() )
        throw "Error when reading steam_avi";
    for( ; ; )
    {
        cv::Mat frame;
        capture >> frame;
        if(frame.empty())
            break;
        frame = frame(cv::Rect(frame.size().width/2,0,frame.size().width/2, frame.size().height));

        if(!once){
            op::Array<float> opOutput = op.forward(frame);
            Eigen::MatrixXf opOutputEigen = convertOPtoEigen(opOutput);

            // Renderer
            glfwInit();
            params.cameraSize = frame.size();
            params.fl = 500.;
            params.cx = params.cameraSize.width/2;
            params.cy = params.cameraSize.height/2;
            params.tx = 0;
            params.ty = 0;
            params.tz = 0;
            params.rx = 0;
            params.ry = 0;
            params.rz = 0;
            renderer.setCameraParams(params);
            renderer.startOnThread("thread");
            once = true;

            // Noise
            Eigen::MatrixXf noiseVector(85,1);
            Eigen::MatrixXf initialVal(85,2);
            for(int i=0; i<24; i++){
                initialVal(i*3 + 0, 0) = pf.meanSMPL.mPose(i,0);
                initialVal(i*3 + 1, 0) = pf.meanSMPL.mPose(i,1);
                initialVal(i*3 + 2, 0) = pf.meanSMPL.mPose(i,2);
            }
            for(int i=0; i<10; i++){
                initialVal(72+i,0) = pf.meanSMPL.mBetas(i,0);
            }
            for(int i=0; i<3; i++){
                initialVal(82+i,0) = pf.meanSMPL.mTrans(i,0);
            }
            for(int i=0; i<72; i++){
                //initialVal(i,0) += 0.1; // Add noise
                initialVal(i,1) = 0.01;
                noiseVector(i,0) = 0.02;
            }
            for(int i=72; i<82; i++){
                //initialVal(i,0) = 0;
                initialVal(i,1) = 0.01;
                noiseVector(i,0) = 0.005;
            }
            for(int i=82; i<85; i++){
                //initialVal(i,0) = 0;
                initialVal(i,1) = 0.01;
                noiseVector(i,0) = 0.01;
            }
            initialVal(0,0) = 3.14;
            //initialVal(1,0) = -0.14;
            //initialVal(2,0) = 2.12;
            //initialVal(82,0) = -0.5;
            initialVal(83,0) = 0.2;
            initialVal(84,0) += 3.2;
            pf.initGauss(initialVal);
            pf.setNoise(noiseVector);

            opOutput = op.forward(frame);
            opOutputEigen = convertOPtoEigen(opOutput);

            cout << opOutput << endl;

            pf.input = frame.clone();
            for(int i=0; i<100; i++){
                pf.update();
                cv::Mat debugImg = pf.weightFunction(opOutputEigen, params);

                pf.resampleParticles();
                pf.computeMeanSMPL();
                cv::Mat out = renderer.draw(pf.meanSMPL.mVTemp2, pf.meanSMPL.mF);
                overlayImage(&debugImg, &out, cv::Point());
                cv::imshow("out",debugImg);
                cv::waitKey(15);
            }

            continue;
        }

        //cv::Mat debugImg = frame.clone();
        begin = std::chrono::steady_clock::now();
        op::Array<float> opOutput = op.forward(frame);
        Eigen::MatrixXf opOutputEigen = convertOPtoEigen(opOutput);
        end= std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;

        pf.input = frame.clone();
        pf.update();
        cv::Mat debugImg = pf.weightFunction(opOutputEigen, params);

        pf.resampleParticles();
        pf.computeMeanSMPL();

        cv::Mat out = renderer.draw(pf.meanSMPL.mVTemp2, pf.meanSMPL.mF);
        overlayImage(&debugImg, &out, cv::Point());

        cv::imshow("out",debugImg);
        cv::waitKey(15);
    }
    cv::waitKey(0); // key press to close window
}

void testPF(){
    std::chrono::steady_clock::time_point begin, end;

    // OP
    cv::Mat im1 = cv::imread("/home/ryaadhav/smpl_cpp/data/00001_image.png");
    cv::resize(im1, im1, cv::Size(0,0),3,3);
    OpenPose op;
    op::Array<float> opOutput = op.forward(im1);
    Eigen::MatrixXf opOutputEigen = convertOPtoEigen(opOutput);

    // Renderer
    glfwInit();
    Renderer renderer;
    Renderer::Params params;
    params.cameraSize = im1.size();
    params.fl = 19800.;
    params.cx = params.cameraSize.width/2;
    params.cy = params.cameraSize.height/2;
    params.tx = 0;
    params.ty = 0;
    params.tz = 0;
    params.rx = 0;
    params.ry = 0;
    params.rz = 0;
    renderer.setCameraParams(params);
    renderer.startOnThread("thread");

    SMPLTracker pf(600,85);
    pf.input = im1.clone();
    pf.meanSMPL.loadPoseFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/00001_body.json");
    pf.meanSMPL.updateModel();

    // Noise
    Eigen::MatrixXf noiseVector(85,1);
    Eigen::MatrixXf initialVal(85,2);

    for(int i=0; i<24; i++){
        cout <<  pf.meanSMPL.mPose.row(i) << endl;
        initialVal(i*3 + 0, 0) = pf.meanSMPL.mPose(i,0);
        initialVal(i*3 + 1, 0) = pf.meanSMPL.mPose(i,1);
        initialVal(i*3 + 2, 0) = pf.meanSMPL.mPose(i,2);
    }
    for(int i=0; i<10; i++){
        initialVal(72+i,0) = pf.meanSMPL.mBetas(i,0);
    }
    for(int i=0; i<3; i++){
        initialVal(82+i,0) = pf.meanSMPL.mTrans(i,0);
    }

    for(int i=0; i<72; i++){
        initialVal(i,0) += 0.1; // Add noise
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.01;
    }
    for(int i=72; i<82; i++){
        //initialVal(i,0) = 0;
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.01;
    }
    for(int i=82; i<85; i++){
        //initialVal(i,0) = 0;
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.005;
    }
    initialVal(0,0) += 0.3;
    //initialVal(0,0) += 3.14/2;
    //initialVal(82,0) = 0.02;
    //initialVal(83,0) = 0.08;
    initialVal(84,0) += 0.5;
    pf.initGauss(initialVal);
    pf.setNoise(noiseVector);

    while(1){

        begin = std::chrono::steady_clock::now();
        pf.update();

        //cv::Mat debugImg = pf.input.clone();
        cv::Mat debugImg = pf.weightFunction(opOutputEigen, params);

        end= std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;


        pf.resampleParticles();
        pf.computeMeanSMPL();

        //        cout << pf.computeMean() << endl;
        //        exit(-1);


        cv::Mat out = renderer.draw(pf.meanSMPL.mVTemp2, pf.meanSMPL.mF);
        overlayImage(&debugImg, &out, cv::Point());

        cv::imshow("out",debugImg);
        cv::waitKey(15);
    }

    //exit(-1);

}

void testFit(){
    // Read Image
    cv::Mat im1 = cv::imread(std::string(CMAKE_CURRENT_SOURCE_DIR)+"/data/00001_image.png");
    cv::resize(im1, im1, cv::Size(0,0),3,3);

    // Load SMPL Pose
    SMPL smpl;
    smpl.loadModelFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/male_model.json");
    std::cout.setstate(std::ios_base::failbit);
    smpl.updateModel();
    std::cout.clear();
    smpl.loadPoseFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/00001_body.json");
    smpl.updateModel();

    // Render
    glfwInit();
    Renderer renderer;
    Renderer::Params params;
    params.cameraSize = im1.size();
    params.fl = 19800.;
    params.cx = params.cameraSize.width/2;
    params.cy = params.cameraSize.height/2;
    params.tx = 0;
    params.ty = 0;
    params.tz = 0;
    params.rx = 0;
    params.ry = 0;
    params.rz = 0;
    renderer.setCameraParams(params);
    renderer.startOnThread("thread");
    cv::Mat out = renderer.draw(smpl.mVTemp2, smpl.mF);

    overlayImage(&im1, &out, cv::Point());

    cout << smpl.mTrans << endl;

    while(1){
        cv::imshow("out",im1);
        cv::waitKey(15);
    }
}

int main(int argc, char *argv[])
{
    std::chrono::steady_clock::time_point begin, end;

    testPF();

    return 0;
}
