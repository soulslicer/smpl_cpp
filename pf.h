#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <time.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>

using namespace std;

class ParticleFilter
{
public:

    struct Probability{
        float std;
        float a,b;

        float error;
        float log;
        float pdf;
        float cdf;
        float prob;

        Probability(){}

        Probability(float std){
            this->std = std;
            this->a = (-std::log(std::sqrt(2*M_PI)*this->std));
            this->b = (-0.5/std::pow(this->std,2));
        }

        Probability(float std, float error) {
            this->std = std;
            this->error = error;
            this->a = (-std::log(std::sqrt(2*M_PI)*this->std));
            this->b = (-0.5/std::pow(this->std,2));
            this->log = (this->a + this->b * (pow(fabs(this->error),2)));
            this->pdf = exp(log);
            this->cdf = (0.5*(1+erf(this->error/(this->std*std::sqrt(2)))));
            this->prob = (1-this->cdf)*2;
        }

        Probability getProbability(float error){
            this->error = error;
            Probability probability;
            probability.error = this->error;
            probability.std = this->std;
            probability.a = this->a;
            probability.b = this->b;
            probability.log = (probability.a + probability.b * (pow(fabs(probability.error),2)));
            probability.pdf = exp(probability.log);
            probability.cdf = (0.5*(1+erf(probability.error/(probability.std*std::sqrt(2)))));
            probability.prob = (1-probability.cdf)*2;
            return probability;
        }
    };

    struct Convergence{
        float statePosStd;
        float stateSizeStd;
        float stateOrStd;
        float queuePosStd;
        float queueSizeStd;
        float queueOrStd;
        Eigen::VectorXf stateDiag;
        Eigen::VectorXf queueDiag;
        void print(){
            cout <<
            "S-PSO: " << statePosStd << ", " << stateSizeStd << ", " << stateOrStd << " : " <<
            "Q-PSO: " << queuePosStd << ", " << queueSizeStd << ", " << queueOrStd << " : " <<
             endl;
        }
    };

protected:
    std::default_random_engine rng_;

    Eigen::MatrixXf stateMatrix_;
    Eigen::MatrixXf weightVector_;
    Eigen::MatrixXf rangeMatrix_;
    Eigen::MatrixXf noiseVector_;

    Eigen::MatrixXf generateUniform(float a, float b);
    Eigen::MatrixXf generateGaussian(float mean, float std, int param);
    Eigen::MatrixXf generateGaussian(Eigen::MatrixXf mean, float std, int param);

    float probability_ = 0.0;
    int particleCount_ = 0;
    int paramsCount_ = 0;
    int iterations_ = 0;
    bool forceRange_ = false;

public:
    explicit
    ParticleFilter(int particleCount, int paramsCount);
    void setParticleCount(int count);
    void setParamsCount(int count);
    void setRange(Eigen::MatrixXf rangeMatrix);
    void setNoise(Eigen::MatrixXf noiseVector);
    void reshape();
    void initGauss(Eigen::MatrixXf input);
    void update();
    void resampleParticles();

    Eigen::MatrixXf computeMean();

//    void setLimitVector(MeanVector limitVector);
//    void initializeGaussian(Eigen::Matrix4f transformMatrix, Eigen::Vector3f size,  MeanVector uncertainVector);
//    void initializeGaussian(Eigen::Matrix4f transformMatrix, Eigen::Vector3f size, float posNoise, float sizeNoise, float oNoise);
//    void initialize(float x, float y, float z, float xl, float yl, float zl, float roll, float pitch, float yaw, float posOff, float sizeOff, float oOff);
//    void setNoise(float posNoise_, float sizeNoise_, float oNoise_);
//    float getPositionNoise();
//    float getSizeNoise();
//    float getOrientationNoise();
//    void updateState();
//    void projectState();
//    void clampPose(Eigen::Matrix4f transformMatrix, Eigen::Vector3f size, MeanVector rangeVector, bool rejectionMode);
//    void setParticleCount(int count);
//    void setParticleCountWithReinitialization(int count);
//    int getParticleCount();
//    void resampleParticles();
//    float getProbability();
//    Eigen::MatrixXf getState();
//    MeanVector updateWeights();
//    MeanVector computeMean();
//    MeanVector computeMap();
//    CovarianceMatrix computeStateCovariance();
//    CovarianceMatrix computeQueueCovariance();
//    float computeConvPDF();
//    PointCloudExt<pcl::PointXYZ> getStateCloud();
//    Eigen::Matrix4f getMeanTransform();
//    Eigen::Vector3f getMeanSize();
//    std::vector<float> getWeightVector();
//    Convergence getConvergence();

    //virtual ParticleFilter::Probability costFunction(Eigen::Matrix4f transformMatrix, Eigen::Vector3f size);
    //virtual std::vector<ParticleFilter::Probability> weightFunction() = 0;
};

#endif
