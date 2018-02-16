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

class SMPL{
public:
    Eigen::MatrixXf mV, mVTemp1, mVTemp2;
    Eigen::MatrixXf mF;
    Eigen::MatrixXf mPose;
    Eigen::MatrixXf mKintreeTable;
    Eigen::MatrixXf mJ, mJTemp1;
    Eigen::MatrixXf mTrans;
    Eigen::MatrixXf mWeights;
    Eigen::MatrixXf mWeightsT;
    Eigen::MatrixXf vertSymIdxs;
    Eigen::MatrixXf mBetas;
    typedef Eigen::Matrix<float, 4, 24> BlockMatrix;
    std::vector<Eigen::MatrixXf> weightedBlockMatrix1;
    std::vector<BlockMatrix> blocks;
    std::vector<Eigen::MatrixXf> mShapeDirs;

    TensorD<3> mShapedDirsTensor;

    SMPL(){
        blocks.resize(4);
        weightedBlockMatrix1.resize(4);
        mBetas.resize(10,1);
        mBetas.setZero();
    }

    enum Part
    {
        BODY,   // 0
        LLEG,   // 1
        RLEG,   // 2
        LTORSO,    // 3
        LKNEE,  // 4
        RKNEE,  // 5
        MTORSO,  // 6
        LFOOT,  // 7
        RFOOT,  // 8
        UTORSO, // 9
        LLFOOT, // 10
        RRFOOT, // 11
        NECK,   // 12
        LSHOULDER,  // 13
        RSHOULDER,  // 14
        HEAD,   // 15
        LSHOULDER2, // 16
        RSHOULDER2, // 17
        LELBOW, // 18
        RELBOW, // 19
        LWRIST, // 20
        RWRIST, // 21
        LFINGERS, // 22
        RFINGERS, // 23
    };

    enum Shape
    {
        S0,
        S1,
        S2,
        S3,
        S4,
        S5,
        S6,
        S7,
        S8,
        S9
    };

    void setPose(Part part, Eigen::Vector3f vec){
        int row = -1;
        if(part == Part::BODY) row = 0;
        if(part == Part::LLEG) row = 1;
        if(part == Part::RLEG) row = 2;
        if(part == Part::LTORSO) row = 3;
        if(part == Part::LKNEE) row = 4;
        if(part == Part::RKNEE) row = 5;
        if(part == Part::MTORSO) row = 6;
        if(part == Part::LFOOT) row = 7;
        if(part == Part::RFOOT) row = 8;
        if(part == Part::UTORSO) row = 9;
        if(part == Part::LLFOOT) row = 10;
        if(part == Part::RRFOOT) row = 11;
        if(part == Part::HEAD) row = 12;
        if(part == Part::LSHOULDER) row = 13;
        if(part == Part::RSHOULDER) row = 14;
        if(part == Part::NECK) row = 15;
        if(part == Part::LSHOULDER2) row = 16;
        if(part == Part::RSHOULDER2) row = 17;
        if(part == Part::LELBOW) row = 18;
        if(part == Part::RELBOW) row = 19;
        if(part == Part::LWRIST) row = 20;
        if(part == Part::RWRIST) row = 21;
        if(part == Part::LFINGERS) row = 22;
        if(part == Part::RFINGERS) row = 23;
        mPose.row(row) = vec;
    }

    void setShape(Shape shape, float val){
        int row = -1;
        if(shape == Shape::S0) row = 0;
        if(shape == Shape::S1) row = 1;
        if(shape == Shape::S2) row = 2;
        if(shape == Shape::S3) row = 3;
        if(shape == Shape::S4) row = 4;
        if(shape == Shape::S5) row = 5;
        if(shape == Shape::S6) row = 6;
        if(shape == Shape::S7) row = 7;
        if(shape == Shape::S8) row = 8;
        if(shape == Shape::S9) row = 9;
        mBetas(row) = val;
    }

    bool loadTensorFromJSON(const Json::Value& json, TensorD<3>& t, bool debug = false){
        int depth = json.size();
        int rows = json[0].size();
        int cols = json[0][0].size();
        if(debug){
            cout << "D: " << depth;
            cout << " R: " << rows;
            cout << " C: " << cols << endl;
        }

        t.resize({depth,rows,cols});

        for(int d=0; d<depth; d++){
            for(int r=0; r<rows; r++){
                for(int c=0; c<cols; c++){
                    t(d,r,c) = json[d][r][c].asFloat();
                }
            }
        }

        return true;
    }

    bool loadEigenVecFromJSON(const Json::Value& json, std::vector<Eigen::MatrixXf>& t, bool debug = false){
        int depth = json.size();
        int rows = json[0].size();
        int cols = json[0][0].size();
        if(debug){
            cout << "D: " << depth;
            cout << " R: " << rows;
            cout << " C: " << cols << endl;
        }

        t.resize(depth);
        for(Eigen::MatrixXf& m : t){
            m.resize(rows, cols);
        }

        for(int d=0; d<depth; d++){
            Eigen::MatrixXf& m = t[d];
            for(int r=0; r<rows; r++){
                for(int c=0; c<cols; c++){
                    t[d](r,c) = json[d][r][c].asFloat();
                }
            }
        }

        return true;
    }

    bool loadEigenFromJSON(const Json::Value& json, Eigen::MatrixXf& m, bool debug = false){
        // Set Shape
        int rows = json.size();
        if(!rows) { cerr << "Matrix Has no Rows" << endl; return false;}
        int cols = json[0].size();
        if(rows == 0) rows = 1;
        m.resize(rows, cols);

        // Load Data
        if(rows > 1){
            for(int i=0; i<rows; i++){
                for(int j=0; j<cols; j++){
                    m(i,j) = json[i][j].asFloat();
                }
            }
        }else{
            throw std::runtime_error("Something wrong");
        }

        return true;
    }

    bool loadModelFromJSONFile(std::string filePath){
        ifstream in(filePath);
        Json::Value root;
        in >> root;
        if(!root.size()){
            cerr << "Failed to load model file" << endl;
            return false;
        }

        cout << root["pose_training_info"] << endl;

        loadEigenFromJSON(root["pose"], mPose);

        loadEigenVecFromJSON(root["shapedirs"], mShapeDirs);

        loadTensorFromJSON(root["shapedirs"], mShapedDirsTensor);

        loadEigenFromJSON(root["f"], mF);

        loadEigenFromJSON(root["kintree_table"], mKintreeTable);

        loadEigenFromJSON(root["J"], mJ);
        mJTemp1 = mJ;

        loadEigenFromJSON(root["trans"], mTrans);

        loadEigenFromJSON(root["v_posed"], mV);
        mV.conservativeResize(mV.rows(),mV.cols()+1);
        mV.col(mV.cols()-1) = Eigen::VectorXf::Ones(mV.rows());
        mVTemp1 = mV;
        mVTemp2 = mV;
        for(Eigen::MatrixXf& w : weightedBlockMatrix1) w.resize(4,mV.rows());

        loadEigenFromJSON(root["weights"], mWeights);
        mWeightsT = mWeights.transpose();

        loadEigenFromJSON(root["vert_sym_idxs"], vertSymIdxs);

        return true;
    }

    Eigen::Matrix4f rod(const Eigen::Vector3f& v, const Eigen::Vector3f& t){
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

    bool updateModel(){

        // Create parent link table
        // {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
        std::map<int,int> parent;
        for(int i=1; i<mKintreeTable.cols(); i++){
            int key = mKintreeTable(1,i); int val = mKintreeTable(0,i);
            parent[key] = val;
        }

        std::vector<Eigen::Matrix4f> globalTransforms(24);
        std::vector<Eigen::Matrix4f> transforms(24);

        // Shape
        TensorD<3> AB = mShapedDirsTensor.dot(mBetas);
        for(int i=0; i<mVTemp1.rows(); i++){
            mVTemp1(i,0) = mV(i,0) + AB(i,0,0);
            mVTemp1(i,1) = mV(i,1) + AB(i,1,0);
            mVTemp1(i,2) = mV(i,2) + AB(i,2,0);
            mVTemp1(i,3) = 1;
        }

        // Body pose
        Eigen::Matrix4f& bodyPose = globalTransforms[0];
        bodyPose = rod(mPose.row(0), mJ.row(0));

        // Global Transforms
        for(int i=1; i<globalTransforms.size(); i++){
            Eigen::Matrix4f& pose = globalTransforms[i];
            pose = globalTransforms[parent[i]] * rod(mPose.row(i), mJ.row(i) - mJ.row(parent[i]));
            mJTemp1(i,0) = pose(0,3);
            mJTemp1(i,1) = pose(1,3);
            mJTemp1(i,2) = pose(2,3);
        }

        // Transforms
        for(int i=0; i<transforms.size(); i++){
            Eigen::Matrix4f& pose = transforms[i];
            Eigen::Vector4f jZero;
            jZero << mJ(i,0), mJ(i,1), mJ(i,2), 0;
            Eigen::Vector4f fx = globalTransforms[i] * jZero; // Only apply rot to jVector
            Eigen::Matrix4f pack = Eigen::Matrix4f::Zero();
            pack(0,3) = fx(0);
            pack(1,3) = fx(1);
            pack(2,3) = fx(2);
            pose = globalTransforms[i] - pack; // Only minus t component from transform with rotated jVector
        }

        // Generate transform from weights
        for(int b=0; b<4; b++){
            BlockMatrix& block = blocks[b];
            for(int i=0; i<24; i++){
                block.col(i) = transforms[i].row(b);
            }
            weightedBlockMatrix1[b] = block*mWeightsT; // Column x VSize ~2ms
        }

        // Transform vertices with weight matrix
        for(int b=0; b<4; b++){
            Eigen::MatrixXf& block = weightedBlockMatrix1[b];
            for(int i=0; i<mV.rows(); i++){
                mVTemp2(i,b) = mVTemp1.row(i) * block.col(i);
            }
        }

        // Need to set final transform too!

        return true;
    }
};

//const int renderWidth = 640;
//const int renderHeight = 480;

//int main(){

//    SMPL smpl;
//    smpl.loadModelFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/model.json");
//    std::cout.setstate(std::ios_base::failbit);
//    smpl.updateModel();
//    std::cout.clear();

//    // OpenGL - Initialization
//    std::cout << "Initializing.." << std::endl;
//    char *myargv [1];
//    int myargc=1;
//    myargv[0]=strdup ("Window");
//    glutInit(&myargc, myargv);
//    glutInitDisplayMode(GLUT_DOUBLE); // glEndList();Enable double buffered mode
//    glutInitWindowSize(renderWidth, renderHeight);   // Set the window's initial width & height
//    glutInitWindowPosition(50, 50); // Position the window's initial top-left corner
//    glutCreateWindow("Window");          // Create window with the given title
//    //glutHideWindow();


//    glEnable(GL_LIGHTING);

//    glEnable( GL_DEPTH_TEST );
//    glShadeModel( GL_SMOOTH );
//    glEnable( GL_CULL_FACE );
//    glClearColor( 1, 1, 1, 1 );

//    glewInit();


//    struct Vertex{
//      GLfloat position[3];
//      GLfloat normal[3];
//      GLfloat texcoord[2];
//    };
//    std::vector<Vertex> vertexdata;
////    Vertex a,b,c,d;
////    a.position[0] = 0;
////    a.position[1] = 0;
////    a.position[2] = 0;
////    b.position[0] = -1;
////    b.position[1] = -1;
////    b.position[2] = 0;
////    c.position[0] = 1;
////    c.position[1] = -1;
////    c.position[2] = 0;
////    d.position[0] = 1.5;
////    d.position[1] = 1.5;
////    d.position[2] = 0;
////    vertexdata.push_back(a);
////    vertexdata.push_back(b);
////    vertexdata.push_back(c);
////    vertexdata.push_back(d);

//    // Iterate v
//    for(int i=0; i<smpl.mVTemp2.rows(); i++){
//        Vertex v;
//        for(int j=0; j<smpl.mVTemp2.cols(); j++){
//            v.position[j] = smpl.mVTemp2(i,j);
//            v.normal[j] = 1;
//        }
//        vertexdata.push_back(v);
//    }

//    std::vector<GLushort> indexdata;

//    // Iterate f
//    for(int i=0; i<smpl.mF.rows(); i++){
//        for(int j=0; j<smpl.mF.cols(); j++){
//            indexdata.push_back(smpl.mF(i,j));
//        }
//    }

//    // Create and bind a VAO
//    GLuint vao;
//    glGenVertexArrays(1, &vao);
//    glBindVertexArray(vao);

//    // Create and bind a BO for vertex data
//    GLuint vbuffer;
//    glGenBuffers(1, &vbuffer);
//    glBindBuffer(GL_ARRAY_BUFFER, vbuffer);

//    // copy data into the buffer object
//    glBufferData(GL_ARRAY_BUFFER, vertexdata.size() * sizeof(Vertex), &vertexdata[0], GL_STATIC_DRAW);

//    // set up vertex attributes
//    glEnableVertexAttribArray(0);
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position)); // vertices
//    glEnableVertexAttribArray(1);
//    glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, sizeof(Vertex), (void*)offsetof(Vertex, normal)); // normals
//    glEnableVertexAttribArray(2);
//    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texcoord)); // normals

//    // Create and bind a BO for index data
//    GLuint ibuffer;
//    glGenBuffers(1, &ibuffer);
//    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuffer);

//    // copy data into the buffer object
//    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexdata.size() * sizeof(GLushort), &indexdata[0], GL_STATIC_DRAW);

//    glBindVertexArray(0);


//    while(1){
//        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//        glEnable(GL_LIGHTING);
//        glShadeModel( GL_SMOOTH );
//        glEnable( GL_TEXTURE_2D );

//        glViewport( 0, 0, (float)renderWidth/1, (float)renderHeight/1. );
//        glMatrixMode( GL_PROJECTION );
//        glLoadIdentity();
//        gluPerspective( 60, (float)renderWidth/(float)renderHeight, 0.1, 10000. );
//        glMatrixMode( GL_MODELVIEW );
//        glLoadIdentity();

//        //glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
//        glPushMatrix();
//        glTranslatef(0,-0.5,-0.5);
////        glBindVertexArray(vao);
////        glDrawElements(GL_TRIANGLES, indexdata.size(), GL_UNSIGNED_SHORT,     (void*)0);

//        for(int i=0; i<smpl.mF.rows(); i++){
//            glBegin(GL_POLYGON);
//            for(int j=0; j<smpl.mF.cols(); j++){
//                indexdata.push_back(smpl.mF(i,j)+1);

//            }
//            glVertex3fv((const GLfloat*)&vertexdata[smpl.mF(i,0)].position);
//            glNormal3fv((const GLfloat*)&vertexdata[smpl.mF(i,0)].normal);
//            glVertex3fv((const GLfloat*)&vertexdata[smpl.mF(i,1)].position);
//            glNormal3fv((const GLfloat*)&vertexdata[smpl.mF(i,1)].normal);
//            glVertex3fv((const GLfloat*)&vertexdata[smpl.mF(i,2)].position);
//            glNormal3fv((const GLfloat*)&vertexdata[smpl.mF(i,2)].normal);

//            glEnd();
//        }

//        glPopMatrix();

//        glutSwapBuffers();
//        glutPostRedisplay();
//        std::this_thread::sleep_for(std::chrono::milliseconds(10));
//        glutMainLoopEvent();
//    }

//    //        for(int i=0; i<smpl.mF.rows(); i++){
//    //            glBegin(GL_POLYGON);
//    //            for(int j=0; j<smpl.mF.cols(); j++){
//    //                indexdata.push_back(smpl.mF(i,j)+1);

//    //            }
//    //            glVertex3fv((const GLfloat*)&vertexdata[smpl.mF(i,0)].position);
//    //            glVertex3fv((const GLfloat*)&vertexdata[smpl.mF(i,1)].position);
//    //            glVertex3fv((const GLfloat*)&vertexdata[smpl.mF(i,2)].position);
//    //            glEnd();
//    //        }

//}

int main(int argc, char *argv[])
{
    //testTensor();
    //exit(-1);
    bool active = false;
    bool track = false;
    bool joints = false;

    SMPL smpl;
    smpl.loadModelFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/model.json");
    std::cout.setstate(std::ios_base::failbit);
    smpl.updateModel();
    std::cout.clear();

    smpl.setPose(SMPL::LLEG, Eigen::Vector3f(M_PI/180. * 20, M_PI/180. * 20, M_PI/180. * 20));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    smpl.updateModel();
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;

    // Trackbar
    DoubleTrackManager trackManager;
    if(track){
        trackManager.setupWindow("Track");
        trackManager.addTrack("LLEG_X",M_PI);
        trackManager.addTrack("LLEG_Y",M_PI);
        trackManager.addTrack("LLEG_Z",M_PI);
    }

    begin = std::chrono::steady_clock::now();
    op::WRender3D render;
    render.initializationOnThread();
    end= std::chrono::steady_clock::now();
    std::cout << "Time difference Setup = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;
    std::shared_ptr<op::WObject> wObject1 = std::make_shared<op::WObject>();
    //wObject1->loadOBJFile("/home/raaj/project/","hello_smpl.obj","");
    wObject1->loadEigenData(smpl.mVTemp2, smpl.mF);
    wObject1->print();
    render.addObject(wObject1);
    //wObject1->rebuild(op::WObject::RENDER_NORMAL);
    //wObject1->rebuildVArr(op::WObject::RENDER_NORMAL);

    std::shared_ptr<op::WObject> wObject2 = std::make_shared<op::WObject>();
    if(joints){
        Eigen::MatrixXf empt;
        wObject2->loadEigenData(smpl.mJTemp1, empt);
        render.addObject(wObject2);
        wObject2->rebuild(op::WObject::RENDER_POINTS, 10);
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
            smpl.setPose(SMPL::LLEG, Eigen::Vector3f(M_PI/180. * currAng, M_PI/180. * currAng, 0));
            smpl.setPose(SMPL::NECK, Eigen::Vector3f(0, 0, M_PI/180. * currAng));
            smpl.setShape(SMPL::S3, currB);

            begin = std::chrono::steady_clock::now();
            smpl.updateModel();
            wObject1->loadEigenData(smpl.mVTemp2, smpl.mF);
            wObject1->rebuildVArr(op::WObject::RENDER_NORMAL);
            end= std::chrono::steady_clock::now();
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;
        }

        render.workOnThread();

        if(track){
            trackManager.spin();
            if(trackManager.changeOccured()){
                smpl.setPose(SMPL::LLEG, Eigen::Vector3f(trackManager.getTrackValue("LLEG_X"), trackManager.getTrackValue("LLEG_Y"), trackManager.getTrackValue("LLEG_Z")));
                smpl.updateModel();
                wObject1->loadEigenData(smpl.mVTemp2, smpl.mF);
                wObject1->rebuild(op::WObject::RENDER_NORMAL);
            }
        }

    }
}
