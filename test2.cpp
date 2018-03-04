#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <pf.h>
#include <mutex>
#include <condition_variable>
#include <smpl.h>
#include <numeric>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision


#include <renderer.hpp>



int main(){
    glfwInit();
    //glewInit();

    // Load SMPL
    SMPL smpl;
    smpl.loadModelFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/male_model.json");
    std::cout.setstate(std::ios_base::failbit);
    smpl.updateModel();
    std::cout.clear();

    smpl.loadPoseFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/00001_body.json");
    smpl.updateModel();

    // Params
    int totalThreads = 10;
    int totalOps = 300;
    std::chrono::steady_clock::time_point begin, end;
    Renderer::Params params;
    params.cameraSize = cv::Size(640,480);
    params.fl = 19800.;
    params.cx = params.cameraSize.width/2;
    params.cy = params.cameraSize.height/2;
    params.tx = 0;
    params.ty = 0;
    params.tz = 0;
    params.rx = 0;
    params.ry = 0;
    params.rz = 0;

//    // Single thread
//    Renderer r;
//    r.drawNormal = false;
//    r.drawContour = true;
//    r.startOnThread("win");
//    r.setCameraParams(params);
//    cv::Mat out = r.draw(smpl.mVTemp2, smpl.mF);
//    begin = std::chrono::steady_clock::now();
//    for(int i=0; i<1; i++){
//        r.draw(smpl.mVTemp2, smpl.mF);
//    }
//    end= std::chrono::steady_clock::now();
//    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;
//    // Contour
//    begin = std::chrono::steady_clock::now();
//    cv::Mat convert;
//    cv::cvtColor(out, out, CV_BGRA2GRAY);
//    std::vector<std::vector<cv::Point>> contours;
//    RendererManager::findContoursCV(out, contours);
//    cv::drawContours(out, contours, -1, cv::Scalar(255));
//    end= std::chrono::steady_clock::now();
//    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;
//    while(1){
//        cv::imshow("im", out);
//        cv::waitKey(15);
//    }

    // Start Threads
    RendererManager manager;
    for(int i=0; i<totalThreads; i++){
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        manager.addThread(params);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    for(int x=0; x<10; x++){
        begin = std::chrono::steady_clock::now();
        for(int i=0; i<totalOps/totalThreads; i++){
            // Set data
            for(int j=0; j<totalThreads; j++){
                manager.renderParams[j] = params;
                manager.renderDatas[j] = RendererManager::RenderData(&smpl.mVTemp2, &smpl.mF);
            }
            manager.signal();
            manager.wait();
        }
        cout << "**DONE**" << endl;
        cout << manager.renderContours[0].size() << endl;
        //cv::drawContours(manager.renderOutputs[0], manager.renderContours[0], -1, cv::Scalar(255,255,0), 4);

        for(int i=0; i<manager.renderPoints[0].size(); i++){
            cv::circle(manager.renderOutputs[0],manager.renderPoints[0][i], 2, cv::Scalar(255));
        }

        cv::imshow("out",manager.renderOutputs[0]);
        cv::waitKey(15);
        end= std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;
    }
    manager.join();
}


//class ParticleFilterExt : public ParticleFilter{
//public:
//    cv::Mat input;

//    ParticleFilterExt(int particleCount, int paramsCount) : ParticleFilter(particleCount, paramsCount)
//    {
//    }

//    void weightFunction(){
//        ParticleFilter::Probability intensityMeanProb(50);
//        for(int i=0; i<particleCount_; i++){
//            cv::Point pixel(stateMatrix_(0,i),stateMatrix_(1,i));
//            cv::Vec3b colour = input.at<cv::Vec3b>(pixel);
//            weightVector_(i,0) = intensityMeanProb.getProbability(255-colour[0]).log;
//        }
//    }

//    cv::Mat draw(){
//        cv::Mat img = input.clone();
//        for(int i=0; i<particleCount_; i++){
//            cv::Point p(stateMatrix_(0,i),stateMatrix_(1,i));
//            cv::circle(img, p, 1, cv::Scalar(255,0,0),CV_FILLED);
//        }

//        Eigen::MatrixXf mean = computeMean();
//        cv::Point mP(mean(0,0),mean(1,0));
//        cv::circle(img, mP, 5, cv::Scalar(255,255,0),CV_FILLED);
//        return img;
//    }

//};

//int main(){
//    Eigen::MatrixXf a(2,2);
//    a(0,1) = 5;

//    Eigen::MatrixXf b = a;
//    b(0,0) = 6;
//    cout << a << endl;
//    cout << b << endl;
//    exit(-1);

//    ParticleFilterExt pf(600,2);
//    Eigen::MatrixXf noiseVector(2,1);
//    noiseVector << 2,2;
//    Eigen::MatrixXf initialVal(2,2);
//    initialVal <<   320,1,
//                    240,1;
//    Eigen::MatrixXf rangeMatrix(2,2);
//    rangeMatrix <<  0,640,
//                    0,480;
//    pf.setRange(rangeMatrix);
//    pf.setNoise(noiseVector);
//    pf.initGauss(initialVal);



//    int pos = 200;

//    while(1){
//        cv::Mat img(cv::Size(640,480), CV_8UC3, cv::Scalar(0,0,0));
//        cv::circle(img, cv::Point(pos++,200), 70, cv::Scalar(255,255,255),CV_FILLED);
//        cv::Mat mSource_Gray,mDist,mBlobDist;
//        cv::cvtColor(img, mSource_Gray, CV_BGR2GRAY);
//        cv::distanceTransform(mSource_Gray, mDist, CV_DIST_L2, 3);
//        cv::normalize(mDist, mDist, 0, 1., cv::NORM_MINMAX);
//        mDist.convertTo(mDist,CV_8UC1,255,0);
//        cv::cvtColor(mDist,img,CV_GRAY2BGR);

//        pf.input = img.clone();

//        pf.update();
//        pf.weightFunction();
//        pf.resampleParticles();
//        cv::Mat out = pf.draw();
//        cv::imshow("win",out);
//        cv::waitKey(15);
//    }

//    return 0;
//}























