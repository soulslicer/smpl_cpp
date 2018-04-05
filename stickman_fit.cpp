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
    cv::Mat inputImage, debugImg;

    struct Params{
        float fl = 500;
        float cx = 0;
        float cy = 0;
    };
    Params params;

    StickmanTracker(int particleCount) : ParticleFilter(particleCount, 67)
    {
        stickmanFilters.resize(particleCount);
        for(StickMan& stickman : stickmanFilters){
            stickman = meanStickman;
        }

        float minFloat = std::numeric_limits<float>::min();
        float maxFloat = std::numeric_limits<float>::max();
        std::cout << paramsCount_ << std::endl;
        std::cout << totalParams << std::endl;

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
                0.1,1.2,  // LEG KNEE LENGTH
                0.1,1.2,  // KNEE FOOT LENGTH
                0.1,1.2,   // BODY TO FNECK LENGTH
                0.1,1.2,   // SHOULDER WIDTH
                0.1,1.2,  // SHOULDER TO ELBOW
                0.1,1.2,    // ELBOW TO WRIST
                0.1,0.5,  // FNECK TO NECK
                0.05,0.2,   // NOSE HEAD ETC
                -10, 10, //TX
                -10, 10, //TY
                -100, 100; //TZ
        setRange(rangeMatrix);
    }

    Eigen::MatrixXf setStickmanFromPF(StickMan& stickman, Eigen::MatrixXf mean){
        for(int i=0; i<meanStickman.bodyParts; i++){
            stickman.theta.row(i) = Eigen::Vector3f(mean(i*3 + 0, 0), mean(i*3 + 1, 0), mean(i*3 + 2, 0));
        }
        for(int i=0; i<meanStickman.bLengths; i++){
            stickman.beta(i,0) = mean((meanStickman.bodyParts*3) + i, 0);
        }
        for(int i=0; i<3; i++){
            stickman.mTrans(i,0) = mean((meanStickman.bodyParts*3) + meanStickman.bLengths + i, 0);
        }
        return stickman.forward();
    }

    Eigen::MatrixXf computeMeanStickman(){
        Eigen::MatrixXf mean = this->computeMean();
        return setStickmanFromPF(meanStickman, mean);
    }

    cv::Point2i project(Eigen::Vector3f point){
        cv::Point2i pixel;
        pixel.x =(int)(((params.fl*point(0))/point(2)) + params.cx);
        pixel.y =(int)(((params.fl*point(1))/point(2)) + params.cy);
        return pixel;
    }

    float l2distance(const cv::Point2i& a, const cv::Point2i& b){
        return sqrt(pow(a.x-b.x,2) + pow(a.y-b.y,2));
    }

    void drawMeanLines(Eigen::MatrixXf mJ){
        std::vector<cv::Point> pixels(meanStickman.bodyParts);
        for(int j=0; j<meanStickman.bodyParts; j++){
            Eigen::Vector3f hypoPoint(mJ(j,0),mJ(j,1),mJ(j,2));
            pixels[j] = project(hypoPoint);
            cv::circle(debugImg, pixels[j], 3, cv::Scalar(255,255,0),CV_FILLED);
        }

        for (auto& kv : meanStickman.kintree) {
            cv::line(debugImg, pixels[kv.first], pixels[kv.second], cv::Scalar(255,0,0), 3);
        }
    }

    void weightFunction(Eigen::MatrixXf opOutput){
        debugImg = inputImage.clone();

        ParticleFilter::Probability pixelReprojProb(5);
        std::vector<StickMan>& stickmanFiltersSC = stickmanFilters;
        Eigen::MatrixXf weights = Eigen::MatrixXf::Zero(weightVector_.rows(),weightVector_.cols());
#pragma omp parallel for shared(stickmanFiltersSC, weights)
        for(int i=0; i<particleCount_; i++){
            StickMan& stickman = stickmanFiltersSC[i];
            Eigen::MatrixXf mJ = setStickmanFromPF(stickman, stateMatrix_.col(i));

            for(int j=0; j<meanStickman.bodyParts; j++){
                Eigen::Vector3f hypoPoint(mJ(j,0),mJ(j,1),mJ(j,2));
                cv::Point hypoPix = project(hypoPoint);
                //if(hypoPix.x <= 0 || hypoPix.y <=0 || hypoPix.x >= inputImage.size().width || hypoPix.y >= inputImage.size().height ) continue;

                int opIndex = -1;
                //if(j == 1) opIndex = 11;
                if(j == 2) opIndex = 11;
                if(j == 3) opIndex = 8;
                if(j == 4) opIndex = 12;
                if(j == 5) opIndex = 9;
                if(j == 6) opIndex = 13;
                if(j == 7) opIndex = 10;
                if(j == 8) opIndex = 1;
                //if(j == 9) opIndex = 10;
                if(j == 10) opIndex = 0;
                //if(j == 11) opIndex = 1;
                if(j == 12) opIndex = 5;
                if(j == 13) opIndex = 2;
                if(j == 14) opIndex = 6;
                if(j == 15) opIndex = 3;
                if(j == 16) opIndex = 7;
                if(j == 17) opIndex = 4;
                if(opIndex < 0) continue;

                if(!opOutput(opIndex,2)) continue;

                cv::Point truthPix(opOutput(opIndex,0),opOutput(opIndex,1));
                //cv::line(debugImg, hypoPix, truthPix, cv::Scalar(255,0,0));
                //cv::circle(debugImg, hypoPix, 3, cv::Scalar(255,0,0), CV_FILLED);
                float reprojErr = l2distance(truthPix, hypoPix);
                weights(i,0) += pixelReprojProb.getProbability(reprojErr).log;
            }
        }

        weightVector_ = weights;
    }
};

Eigen::MatrixXf convertOPtoEigen(op::Array<float>& opOutput, int person = 0){
    int people = opOutput.getSize()[0];
    Eigen::MatrixXf eigen(opOutput.getSize()[1],opOutput.getSize()[2]);
    for(int r=0; r<eigen.rows(); r++){
        for(int c=0; c<eigen.cols(); c++){
            eigen(r,c) = opOutput[person*eigen.rows()*eigen.cols() + eigen.cols()*r + c];
        }
    }

    return eigen;
}

int main(int argc, char *argv[])
{
    // Idea
    /* Can we encode the error for each body part, in a chained way
     * Depending on that error, we reduce the noise accordingly?
     * If wrist error is small, we reduce noise on bone length wrist and also on wrist-elbow angle
     */

    std::chrono::steady_clock::time_point begin, end;

    // OP
    cv::Mat im1 = cv::imread(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/00001_image.png");
    cv::resize(im1, im1, cv::Size(0,0),3,3);
    OpenPose op;
    op::Array<float> opOutput = op.forward(im1);
    Eigen::MatrixXf opOutputEigen = convertOPtoEigen(opOutput);

    // PF
    StickmanTracker pf(150);
    pf.params.fl = 500.;
    pf.params.cx = im1.size().width/2;
    pf.params.cy = im1.size().height/2;
    pf.inputImage = im1.clone();
    Eigen::MatrixXf noiseVector(pf.totalParams,1);
    Eigen::MatrixXf initialVal = Eigen::MatrixXf::Zero(pf.totalParams,2);
    // Joint Angles all 0
    for(int i=0; i<pf.meanStickman.bodyParts; i++){
        initialVal(i*3 + 0, 0) = 0;
        initialVal(i*3 + 1, 0) = 0;
        initialVal(i*3 + 2, 0) = 0;
    }
    // Specific body part lengths
    initialVal((pf.meanStickman.bodyParts*3)+0,0) = 0.3; // BODY HIP LENGTH
    initialVal((pf.meanStickman.bodyParts*3)+1,0) = 0.2; // HIP WIDTH
    initialVal((pf.meanStickman.bodyParts*3)+2,0) = 0.3; // LEG KNEE LENGTH
    initialVal((pf.meanStickman.bodyParts*3)+3,0) = 0.3; // KNEE FOOT LENGTH
    initialVal((pf.meanStickman.bodyParts*3)+4,0) = 0.2; // BODY TO FNECK LENGTH
    initialVal((pf.meanStickman.bodyParts*3)+5,0) = 0.2; // SHOULDER WIDTH
    initialVal((pf.meanStickman.bodyParts*3)+6,0) = 0.2; // SHOULDER TO ELBOW
    initialVal((pf.meanStickman.bodyParts*3)+7,0) = 0.2; // ELBOW TO WRIST
    initialVal((pf.meanStickman.bodyParts*3)+8,0) = 0.1; // FNECK TO NECK
    initialVal((pf.meanStickman.bodyParts*3)+9,0) = 0.1; // NOSE HEAD ETC
    // Trans
    initialVal((pf.meanStickman.bodyParts*3)+10,0) = 0.0; // X
    initialVal((pf.meanStickman.bodyParts*3)+11,0) = 0.0; // Y
    initialVal((pf.meanStickman.bodyParts*3)+12,0) = 3.0; // Z
    // Noise for Joints
    float mult = 1;
    for(int i=0; i<(pf.meanStickman.bodyParts*3); i++){
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.01*mult;
    }
    // Noise for body part lengths
    for(int i=(pf.meanStickman.bodyParts*3); i<(pf.meanStickman.bodyParts*3)+pf.meanStickman.bLengths; i++){
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.01*mult;
    }
    // Noise for translation
    for(int i=(pf.meanStickman.bodyParts*3)+pf.meanStickman.bLengths; i<(pf.meanStickman.bodyParts*3)+pf.meanStickman.bLengths+3; i++){
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.01*mult;
    }
    initialVal(0,0) += 3.14;
    pf.initGauss(initialVal);
    pf.setNoise(noiseVector);

    // Compute Stickman
    Eigen::MatrixXf mvTemp = pf.computeMeanStickman();
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

    cv::VideoCapture cap(0);

    while(1){

        cap >> im1;
        pf.inputImage = im1.clone();
        if(im1.empty()) break;
        opOutput = op.forward(im1);
        opOutputEigen = convertOPtoEigen(opOutput);

        //Eigen::MatrixXf mVTemp2 = sm.forward();
        begin = std::chrono::steady_clock::now();
        for(int i=0; i<5; i++){
            pf.update();
            pf.weightFunction(opOutputEigen);
            pf.resampleParticles();
            mvTemp = pf.computeMeanStickman();
        }
        end = std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;
        pf.drawMeanLines(mvTemp);

        // NEED A WAY TO CENTER THE 3D MODEL
        // move on body position
        for(int i=1; i<pf.meanStickman.bodyParts; i++){
            mvTemp.row(i) -= mvTemp.row(0);
        }
        mvTemp.row(0) -= mvTemp.row(0);

        render.workOnThread();
        wObject1->clearOBJFile(true);
        wObject1->loadEigenData(mvTemp, mF);
        wObject1->loadKT(pf.meanStickman.kintree);
        wObject1->rebuild(WObject::RENDER_POINTS, 10);

        cv::imshow("win",pf.debugImg);
        cv::waitKey(15);
    }

}
