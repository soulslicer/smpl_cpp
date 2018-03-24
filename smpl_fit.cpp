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
#include <opencv2/flann/miniflann.hpp>

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
            if(opacity == 1) opacity = 0.5;

            for (int c = 0; opacity > 0 && c < src->channels(); ++c)
            {
                unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
                unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
                src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
            }
        }
    }
}

int maxVal(int i, int j, int k){
    int ret = max(i,j);
    ret = max(ret, k);
    return ret;
}

cv::Mat performSobel(cv::Mat& img, int kernel, double scale, double delta)
{
    cv::Mat newImg = img.clone();
    cv::cvtColor(img, newImg, CV_BGR2GRAY);

    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y, grad;
    cv::Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    cv::Sobel( newImg, grad_x, CV_16S, 1, 0, kernel, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    cv::Sobel( newImg, grad_y, CV_16S, 0, 1, kernel, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    return grad;
}

cv::Mat smartEdge(cv::Mat& rgbImage, int convType, int blurSize, int kernelSize, float scale, int intensityThreshold = 0, int valueThreshold = 255){
    cv::Mat convImage;
    if(convType > 0)
        cv::cvtColor(rgbImage, convImage, convType);
    else
        convImage = rgbImage.clone();
    std::vector<cv::Mat> channels(3);
    cv::split(convImage, channels);

    cv::Mat valueImage;
    if(valueThreshold < 255){
        cv::Mat hsvImage;
        cv::cvtColor(rgbImage, hsvImage, CV_BGR2HSV);
        std::vector<cv::Mat> hsvChannels(3);
        cv::split(hsvImage, hsvChannels);
        valueImage = hsvChannels[2].clone();
    }

    std::vector<cv::Mat> sobleChannels(3);
    for(int i=0; i<sobleChannels.size(); i++){
        sobleChannels[i] = channels[i].clone();
        if(blurSize > 0) cv::GaussianBlur(sobleChannels[i], sobleChannels[i], cv::Size(blurSize,blurSize), 0, 0);
        performSobel(sobleChannels[i], kernelSize, scale, 0);
    }

    cv::Mat combinedEdge(rgbImage.rows, rgbImage.cols, CV_8UC1, cv::Scalar(0));
    for(int i = 0;i < combinedEdge.cols;i++){
        for(int j = 0;j < combinedEdge.rows;j++){
            if(valueThreshold < 255){
                int val = valueImage.at<uint8_t>(j,i);
                if(val > valueThreshold) continue;
            }
            int e0 = sobleChannels[0].at<uint8_t>(j,i);
            int e1 = sobleChannels[1].at<uint8_t>(j,i);
            int e2 = sobleChannels[2].at<uint8_t>(j,i);
            if(convType == CV_BGR2HSV) e0 = 0;
            int max = maxVal(e0,e1,e2);
            if(max > intensityThreshold) combinedEdge.at<uint8_t>(j,i) = max;
        }
    }

    return combinedEdge;
}

class PointsWithTree{
public:
    std::vector<cv::Point2f> points;
    cv::flann::KDTreeIndexParams indexParams;
    std::shared_ptr<cv::flann::Index> kdtree;

    PointsWithTree(){

    }

    void setPoints(std::vector<std::vector<cv::Point>>& segContours){
        points.clear();
        for(int i=0; i<segContours.size(); i++){
            for(int j=0; j<segContours[i].size(); j++){
                points.push_back(cv::Point2f(segContours[i][j].x,segContours[i][j].y));
            }
        }
        kdtree = std::shared_ptr<cv::flann::Index>(new cv::flann::Index(cv::Mat(points).reshape(1), indexParams));
    }

    cv::Point nn(cv::Point sample){
        vector<float> query;
        query.push_back(sample.x);
        query.push_back(sample.y);
        vector<int> indices;
        vector<float> dists;
        kdtree->knnSearch(query, indices, dists, 1);
        return cv::Point(points[indices[0]].x,points[indices[0]].y);
    }
};

class SMPLTracker : public ParticleFilter{
public:
    SMPL meanSMPL;
    cv::Mat input, edgeInput;
    const float PI = M_PI;
    const float PI2 = PI*2;
    std::vector<SMPL> smplFilters;
    RendererManager manager;
    int totalThreads = 10;
    bool contourFit = false;

    SMPLTracker(int particleCount, int paramsCount, Renderer::Params params) : ParticleFilter(particleCount, paramsCount)
    {
//        for(int i=0; i<totalThreads; i++){
//            std::this_thread::sleep_for(std::chrono::milliseconds(50));
//            manager.addThread(params);
//        }

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
        //setRange(rangeMatrix);
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

    cv::Mat weightFunction(Eigen::MatrixXf opOutput, Renderer::Params& params, PointsWithTree& contourTree){
        cv::Mat debugImg = input.clone();

        ParticleFilter::Probability pixelReprojProb(5);
        ParticleFilter::Probability intensityProb(30);
        std::vector<SMPL>& smplFiltersSC = smplFilters;
        Eigen::MatrixXf weights = Eigen::MatrixXf::Zero(weightVector_.rows(),weightVector_.cols());
#pragma omp parallel for shared(smplFiltersSC, weights)
        for(int i=0; i<particleCount_; i++){
            SMPL& smpl = smplFiltersSC[i];
            setSMPLFromPF(smpl, stateMatrix_.col(i), !contourFit);

            for(int j=0; j<24; j++){
                Eigen::Vector3f hypoPoint(smpl.mJTemp2(j,0),smpl.mJTemp2(j,1),smpl.mJTemp2(j,2));
                cv::Point hypoPix = project(hypoPoint, params);
                //cv::circle(debugImg, hypoPix, 3, cv::Scalar(0,255,0), CV_FILLED);

                int opIndex = -1;
                if(j == 1) opIndex = 11;
                if(j == 2) opIndex = 8;
                if(j == 4) opIndex = 12;
                if(j == 5) opIndex = 9;
                if(j == 7) opIndex = 13;
                if(j == 8) opIndex = 10;
                if(j == 12) opIndex = 18; // Head
                if(j == 15) opIndex = 1;
                if(j == 16) opIndex = 5;
                if(j == 17) opIndex = 2;
                if(j == 18) opIndex = 6;
                if(j == 19) opIndex = 3;
                if(j == 20) opIndex = 7;
                if(j == 21) opIndex = 4;

                if(opIndex < 0) continue;

                cv::Point truthPix(opOutput(opIndex,0),opOutput(opIndex,1));
                //cv::circle(debugImg, truthPix, 3, cv::Scalar(255,0,0), CV_FILLED);
                cv::line(debugImg, hypoPix, truthPix, cv::Scalar(255,0,0));
                float reprojErr = l2distance(truthPix, hypoPix);
                weights(i,0) += pixelReprojProb.getProbability(reprojErr).log;
            }

        }

        if(contourFit){
            std::vector<std::vector<cv::Point>> contours(particleCount_);
            std::vector<std::vector<std::vector<cv::Point>>> outerContours(particleCount_);
            for(int i=0; i<particleCount_/totalThreads; i++){
                // Set data
                for(int j=0; j<totalThreads; j++){
                    int pIndex = i*totalThreads + j;
                    SMPL& smpl = smplFilters[pIndex];
                    manager.renderParams[j] = params;
                    manager.renderDatas[j] = RendererManager::RenderData(&smpl.mVTemp2, &smpl.mF);
                }
                manager.signal();
                manager.wait();
                for(int j=0; j<totalThreads; j++){
                    int pIndex = i*totalThreads + j;
                    contours[pIndex] = manager.renderPoints[j];
                    outerContours[pIndex] = manager.renderContours[j];
                }
            }
            #pragma omp parallel for shared(contourTree, weights)
            for(int i=0; i<particleCount_; i++){

                for(int m=0; m<outerContours[i].size(); m++){
                    for(int l=0; l<outerContours[i][m].size(); l++){
                        cv::Point closestP = contourTree.nn(outerContours[i][m][l]);
                        float reprojErr = l2distance(outerContours[i][m][l], closestP);
                        //cv::line(debugImg, outerContours[i][m][l], closestP, cv::Scalar(255,0,0));
                        weights(i,0) += pixelReprojProb.getProbability(reprojErr).log;
                    }
                }

//                for(int j=0; j<contours[i].size(); j++){
//                    cv::Point closestP = contourTree.nn(contours[i][j]);
//                    float reprojErr = l2distance(contours[i][j], closestP);
//                    cv::line(debugImg, contours[i][j], closestP, cv::Scalar(255,0,0));
//                    weights(i,0) += pixelReprojProb.getProbability(reprojErr).log;
//                    //char intensity = edgeInput.at<char>(contours[i][j]);
//                    //cv::circle(debugImg, contours[i][j], 1, cv::Scalar(intensity));
//                    //weights(i,0) += intensityProb.getProbability(255-intensity).log;
//                }
            }
            cout << "ok" << endl;
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
    int origSize = opOutput.getSize()[1];
    Eigen::MatrixXf eigen(opOutput.getSize()[1],opOutput.getSize()[2]);
    for(int r=0; r<eigen.rows(); r++){
        for(int c=0; c<eigen.cols(); c++){
            eigen(r,c) = opOutput[eigen.cols()*r + c];
        }
    }

    // Add head
    eigen.conservativeResize(eigen.rows()+1, eigen.cols());
    eigen.row(origSize) = Eigen::Vector3f((eigen(16,0)+eigen(17,0))/2, (eigen(16,1)+eigen(17,1))/2, (eigen(16,2)+eigen(17,2))/2);

    return eigen;
}

void testPF(){
    std::chrono::steady_clock::time_point begin, end;

    // OP
    cv::Mat im1 = cv::imread(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/00001_image.png");
    cv::Mat edge = cv::imread(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/00001_edge.png",0);
    cv::Mat seg = cv::imread(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/im0002_segmentation.png",0);
    cout << im1.size() << endl;    
    cv::resize(im1, im1, cv::Size(0,0),3,3);
    cv::resize(edge, edge, cv::Size(0,0),3,3);
    cv::resize(seg, seg, cv::Size(0,0),3,3);

    std::vector<std::vector<cv::Point>> segContours;
    RendererManager::findContoursCV(seg, segContours);
    PointsWithTree contourPoints;
    contourPoints.setPoints(segContours);

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

    SMPLTracker pf(200,85,params);
    pf.input = im1.clone();
    pf.edgeInput = edge.clone();
    pf.meanSMPL.loadPoseFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/data/00001_body.json");
    pf.meanSMPL.updateModel();

    // Noise
    Eigen::MatrixXf noiseVector(85,1);
    Eigen::MatrixXf initialVal = Eigen::MatrixXf::Zero(85,2);

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
        initialVal(i,0) += 0.1; // Add noise
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.01;
    }
    for(int i=72; i<82; i++){
        //initialVal(i,0) += 0.1; // Noise
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.01;
    }
    for(int i=82; i<85; i++){
        initialVal(i,0) += 0.5; // Noise
        initialVal(i,1) = 0.01;
        noiseVector(i,0) = 0.01;
    }
    //initialVal(72+0,0) -= 3;
    //initialVal(0,0) += 3.14;
    //initialVal(82,0) = 0.02;
    //initialVal(83,0) = 0.08;
    initialVal(84,0) += 0.5;
    pf.initGauss(initialVal);
    pf.setNoise(noiseVector);

    cv::imshow("out",im1);
    cv::waitKey(1000);

    int count = 0;
    while(1){
        count++;
        //if(count > 3) pf.contourFit = true;

        begin = std::chrono::steady_clock::now();
        pf.update();

        //cv::Mat debugImg = pf.input.clone();
        cv::Mat debugImg = pf.weightFunction(opOutputEigen, params, contourPoints);
        end= std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;

        pf.resampleParticles();
        pf.computeMeanSMPL();
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
