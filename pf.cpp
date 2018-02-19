#include "pf.h"
#include <random>

ParticleFilter::ParticleFilter(int particleCount, int paramsCount){
    srand((unsigned int) time(0));
    setParticleCount(particleCount);
    setParamsCount(paramsCount);
    reshape();
}

Eigen::MatrixXf ParticleFilter::generateUniform(float a, float b){
    srand((unsigned int) time(0));
    Eigen::MatrixXf rowVector = (b-a)*((Eigen::MatrixXf::Random(1,particleCount_) + Eigen::MatrixXf::Constant(1,particleCount_,1)) / 2) + Eigen::MatrixXf::Constant(1,particleCount_,1);
    return rowVector;
}

Eigen::MatrixXf ParticleFilter::generateGaussian(float mean, float std, int param){
    Eigen::MatrixXf rowVector = Eigen::MatrixXf::Zero(1,particleCount_);
    std::normal_distribution<float> normalDistribution(mean, std);
    for(int i=0; i<particleCount_; ++i){
        rowVector(0,i) = normalDistribution(rng_);
        if(forceRange_){
            float start = rangeMatrix_(param, 0);
            float end = rangeMatrix_(param, 1);
            int count = 0;
            while(rowVector(0,i) < start || rowVector(0,i) > end){
                rowVector(0,i) = normalDistribution(rng_);
                count++;
                if(count > 500){
                    if(rowVector(0,i) < start) rowVector(0,i) = start;
                    if(rowVector(0,i) > end) rowVector(0,i) = end;
                    break;
                }
            }
        }
    }
    return rowVector;
}

Eigen::MatrixXf ParticleFilter::generateGaussian(Eigen::MatrixXf mean, float std, int param){
    Eigen::MatrixXf rowVector = Eigen::MatrixXf::Zero(1,particleCount_);
    std::normal_distribution<float> normalDistribution(0, std);
    for(int i=0; i<particleCount_; ++i){
        rowVector(0,i) = mean(0,i) + normalDistribution(rng_);
        cout << std << endl;
        if(forceRange_){
            float start = rangeMatrix_(param, 0);
            float end = rangeMatrix_(param, 1);
            int count = 0;
            while(rowVector(0,i) < start || rowVector(0,i) > end){
                rowVector(0,i) = mean(0,i) + normalDistribution(rng_);
                count++;
                if(count > 500){
                    if(rowVector(0,i) < start) rowVector(0,i) = start;
                    if(rowVector(0,i) > end) rowVector(0,i) = end;
                    break;
                }
            }
        }
    }
    return rowVector;
}

void ParticleFilter::setParticleCount(int count){
    particleCount_ = count;
}

void ParticleFilter::setParamsCount(int count){
    paramsCount_ = count;
}

void ParticleFilter::setRange(Eigen::MatrixXf rangeMatrix){
    if(rangeMatrix.cols() != 2 || rangeMatrix.rows() != paramsCount_){
        throw std::runtime_error("Wrong Size for limit vector");
    }
    forceRange_ = true;
    rangeMatrix_= rangeMatrix;
}

void ParticleFilter::reshape(){
    if(particleCount_ == 0 || paramsCount_ == 0){
        throw std::runtime_error("Particle or Params count not set");
    }
    stateMatrix_ = Eigen::MatrixXf::Zero(paramsCount_, particleCount_);
    weightVector_ = Eigen::MatrixXf::Zero(particleCount_, 1);
    rangeMatrix_ = Eigen::MatrixXf::Zero(paramsCount_, 2);
}

void ParticleFilter::initGauss(Eigen::MatrixXf input){
    if(input.rows() != paramsCount_ || input.cols() != 2){
        throw std::runtime_error("Wrong input size");
    }
    for(int i=0; i<paramsCount_; i++){
        stateMatrix_.row(i) = generateGaussian(input(i,0),input(i,1), i);
    }
}

void ParticleFilter::setNoise(Eigen::MatrixXf noiseVector){
    if(noiseVector.rows() != paramsCount_ || noiseVector.cols() != 1){
        throw std::runtime_error("Wrong noise size");
    }
    noiseVector_ = noiseVector;
}

void ParticleFilter::update(){
    for(int i=0; i<paramsCount_; i++){
        stateMatrix_.row(i) = generateGaussian(stateMatrix_.row(i),noiseVector_(i,0), i);
    }
}

Eigen::MatrixXf ParticleFilter::computeMean(){
    Eigen::MatrixXf meanVector = Eigen::MatrixXf(paramsCount_,1);
    for(int i=0;i<paramsCount_;i++){
        meanVector(i) = stateMatrix_.row(i).mean();
    }
    return meanVector;
}

void ParticleFilter::resampleParticles()
{
    iterations_++;

    Eigen::VectorXf weightVectorSubset = weightVector_.col(0);
    Eigen::VectorXf L = (weightVectorSubset - Eigen::VectorXf::Constant(particleCount_, weightVectorSubset.maxCoeff())).array().exp();
    Eigen::VectorXf Q = L / L.sum();
    Eigen::VectorXf R = Eigen::VectorXf::Constant(particleCount_, 0.0);
    R(0,0) = Q(0,0);
    for(int i=1; i<particleCount_; i++)
        R(i,0) = R(i-1,0) + Q(i,0);

    srand((unsigned int) time(0));
    Eigen::VectorXf T = Eigen::VectorXf::Random(particleCount_);
    T = (T + Eigen::VectorXf::Constant(particleCount_, 1)) / 2;

    Eigen::VectorXf I = Eigen::VectorXf::Constant(particleCount_, 0.0);
    for(int i=0; i<particleCount_; i++){
        for(int j=0; j<particleCount_; j++){
            if(R(j,0) >= T(i,0)){
                I(i,0) = j;
                break;
            }
        }
    }

    Eigen::MatrixXf newStateMatrix = Eigen::MatrixXf(stateMatrix_.rows(), stateMatrix_.cols());
    for(int i=0; i<particleCount_; i++){
        newStateMatrix.col(i) = stateMatrix_.col(I(i,0));
    }
    stateMatrix_ = newStateMatrix;
}
