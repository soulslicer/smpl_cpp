#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <iomanip>
#include <numeric>
using namespace std;

template<int T>
class TensorD : public Eigen::Tensor<float, T>{
private:
    int tDim = -1;
public:
    typedef Eigen::Tensor<float, T> Base;
    typedef Eigen::TensorMap<Eigen::Tensor<float, T>> Map;
    typedef Eigen::TensorMap<Eigen::Tensor<float, 2>> Map2D;
    typedef Eigen::TensorMap<Eigen::Tensor<float, 3>> Map3D;
    typedef Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float>, const Base, const Base> BinaryOp;
    typedef Eigen::TensorContractionOp<const std::array<Eigen::IndexPair<int>,1ul>, const Base, const Eigen::Tensor<float, 2>> Contraction2Op;
    typedef Eigen::TensorContractionOp<const std::array<Eigen::IndexPair<int>,1ul>, const Base, const Map2D> Contraction2MapOp;

    TensorD(){
        tDim = T;
    }

    TensorD(const std::array<int, T>& size){
        tDim = size.size();
        resize(size);
    }

    TensorD( const Base &d ) : Base(d)
    {
        tDim = d.dimensions().size();
    }

    TensorD(const BinaryOp& x)
    {
        this->base() = x;
        tDim = this->base().dimensions().size();
    }

    TensorD(const Contraction2Op& x)
    {
        this->base() = x;
        tDim = this->base().dimensions().size();
    }

    TensorD(const Contraction2MapOp& x)
    {
        this->base() = x;
        tDim = this->base().dimensions().size();
    }

    TensorD<T> operator+(TensorD<T>& n){
        return (this->base()+n.base());
    }

    ~TensorD(){

    }

    void resize(const std::array<int, T>& size){
        tDim = size.size();
        this->Base::resize(size);
    }

    void printSize(){
        if(tDim == 3)
            cout << "D: " << this->dimensions()[0] << " R: " << this->dimensions()[1] << " C: " << this->dimensions()[2] << endl;
        else if(tDim == 2)
            cout << "R: " << this->dimensions()[0] << " C: " << this->dimensions()[1] << endl;
        else
            throw std::runtime_error("printSize dim > 3 not implemented. Or failed to call resize");
    }

    int depth(){
        if(tDim != 3) throw std::runtime_error("This matrix has no depth");
        return this->dimensions()[0];
    }

    int rows(){
        if(tDim == 3)
            return this->dimensions()[1];
        else if(tDim == 2)
            return this->dimensions()[0];
    }

    int cols(){
        if(tDim == 3)
            return this->dimensions()[2];
        else if(tDim == 2)
            return this->dimensions()[1];
    }

    void printAtDepth(int d, int rowCount = 4, int colCount = 4){
        if(tDim != 3) throw std::runtime_error("This matrix has no depth");
        std::cout << std::setprecision(3) << std::fixed;
        cout << "[";
        for(int i=0; i<rows(); i++){
            if(i < rowCount || i >= rows()-rowCount){
                cout << "[";
                for(int j=0; j<cols(); j++){
                    if(j < colCount || j >= cols()-colCount)
                        cout << this->operator()(d,i,j) << "     ";
                    else if(j == colCount)
                        cout << " ... ";
                }
                cout << "]" << endl;
            }else if(i == rowCount){
                cout << "..." << endl;
                cout << "..." << endl;
            }
        }
        cout << "]" << endl;
    }

    Eigen::MatrixXf getMatrix(int d){
        Eigen::MatrixXf m;
        m.resize(rows(),cols());
        for(int r=0;r<rows(); r++){
            for(int c=0;c<cols(); c++){
                m(r,c) = this->operator()(d,r,c);
            }
        }
        return m;
    }

    Eigen::MatrixXf getMatrix(){
        Eigen::MatrixXf m;
        m.resize(rows(),cols());
        for(int r=0;r<rows(); r++){
            for(int c=0;c<cols(); c++){
                m(r,c) = this->operator()(r,c);
            }
        }
        return m;
    }

    void setFromMatrix(const Eigen::MatrixXf& m){
        if(!(m.rows() == rows() && m.cols() == cols())) resize({m.rows(), m.cols()});
        for(int r=0;r<rows(); r++){
            for(int c=0;c<cols(); c++){
                this->operator()(r,c) = m(r,c);
            }
        }
    }

    TensorD<3> dot(const TensorD<2>& x){
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(2, 0) };
        return this->base().contract(x.base(), product_dims);
    }

    TensorD<3> dot(Eigen::MatrixXf& m){
        auto map = Map2D(m.data(), m.rows(), m.cols());
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(2, 0) };
        return this->base().contract(map, product_dims);
    }

    void print2D(int rowCount = 4, int colCount = 4);
    void print3D(int depthCount = 4, int rowCount = 4, int colCount = 4);
    void print();
};

template<> void TensorD<2>::print2D(int rowCount, int colCount){
    printSize();
    std::cout << std::setprecision(3) << std::fixed;
    cout << "[";
    for(int i=0; i<rows(); i++){
        if(i < rowCount || i >= rows()-rowCount){
            cout << "[";
            for(int j=0; j<cols(); j++){
                if(j < colCount || j >= cols()-colCount)
                    cout << this->operator()(i,j) << "     ";
                else if(j == colCount)
                    cout << " ... ";
            }
            cout << "]" << endl;
        }else if(i == rowCount){
            cout << "..." << endl;
            cout << "..." << endl;
        }
    }
    cout << "]" << endl;
}

template<> void TensorD<3>::print3D(int depthCount, int rowCount, int colCount){
    printSize();
    std::cout << std::setprecision(3) << std::fixed;
    cout << "[";
    for(int d=0; d<depth(); d++){
        if(d < depthCount || d >= depth()-depthCount){
            printAtDepth(d, rowCount, colCount);
        }else if(d==depthCount){
            cout << "..." << endl;
            cout << "..." << endl;
        }
    }
    cout << "]" << endl;
}

template<> void TensorD<2>::print(){
    print2D();
}

template<> void TensorD<3>::print(){
    print3D();
}

std::string getSpaces(int spaces, bool empt = false){
    std::string s = "";
    if(empt) return s;
    for(int i=0; i<spaces; i++)
        s+=" ";
    return s;
}

void printEigen(const Eigen::MatrixXf& m, int colCount = 4, int rowCount = 4, int spaces = 4){
    std::cout << std::setprecision(3) << std::fixed;
    cout << "R: " << m.rows() << " C: " << m.cols() << endl;
    cout << "[";
    for(int i=0; i<m.rows(); i++){
        if(i < rowCount){
            cout << "[";
            for(int j=0; j<m.cols(); j++){
                bool empt = false;
                if(j == m.cols()-1) empt = true;
                if(j < colCount)
                    cout << m(i,j) << getSpaces(spaces,empt);
                else if(j == colCount)
                    cout << " ... ";
                else if(j >= m.cols()-colCount)
                    cout << m(i,j) << getSpaces(spaces,empt);
            }
            cout << "]" << endl;
        }else if(i == rowCount){
            cout << "..." << endl;
            cout << "..." << endl;
        }else if(i >= m.rows()-rowCount){
            cout << "[";
            for(int j=0; j<m.cols(); j++){
                bool empt = false;
                if(j == m.cols()-1) empt = true;
                if(j < colCount)
                    cout << m(i,j) << getSpaces(spaces,empt);
                else if(j == colCount)
                    cout << " ... ";
                else if(j >= m.cols()-colCount)
                    cout << m(i,j) << getSpaces(spaces,empt);
            }
            cout << "]" << endl;
        }
    }
    cout << "]" << endl;
}


#endif
