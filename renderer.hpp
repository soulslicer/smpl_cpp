#ifndef OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP
#define OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP

#include <mutex>
#include <stdio.h>
#include <GL/glew.h>

#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <GL/freeglut_std.h>
#include <GL/glu.h>
#include <opencv2/opencv.hpp>
#include <glfw3.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <mutex>
#include <functional>
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <thread>
#include <chrono>

#include <mutex>
#include <condition_variable>

using namespace std;
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

struct OBJObject;
struct OBJFace;
struct OBJVertex;
struct OBJNormal;
struct OBJTexture;
struct OBJFaceItem;
struct OBJMaterial;
struct Vertex;

class WObject
{
public:
    const static int RENDER_NORMAL = 0;
    const static int RENDER_POINTS = 1;
    const static int RENDER_WIREFRAME = 2;

    WObject();
    ~WObject();
    bool clearOBJFile(bool clearObject = true);
    void print();
    bool loadOBJFile( const std::string& data_path, const std::string& mesh_filename, const std::string& material_filename );
    bool loadEigenData(const Eigen::MatrixXf& v, const Eigen::MatrixXf& f);
    bool loadKT(std::map<int, int> kintree);
    void render();
    void rebuild(int renderType = WObject::RENDER_NORMAL, float param = 1);
    void rebuildVArr(int renderType = WObject::RENDER_NORMAL, float param = 1);

private:
    std::string mDataPath;
    std::string mCurrentMaterial;
    std::shared_ptr<OBJObject> mObject;
    std::map<std::string,GLuint> textures;
    std::map<int, int> kintree;
    GLuint vao;
    GLuint vbuffer;
    GLuint listId;
    GLuint ibuffer;
    GLuint getTexture(const std::string& filename);
    bool releaseTexture(const std::string& filename);
    bool loadTexture(const std::string& filename, bool clamp);
    void processMaterialLine( const std::string& line );
    void processMeshLine( const std::string& line );
};

// This worker will do 3-D rendering
class  WRender3D
{
public:
    std::mutex renderMutex;

    WRender3D();

    ~WRender3D();

    void initializationOnThread();

    void workOnThread();

    void addObject(std::shared_ptr<WObject> wObject);

};

class Renderer{
public:
    int renderWidth = 640;
    int renderHeight = 480;
    bool drawNormal = true;
    bool drawContour = false;
    bool drawLighting = true;

    const std::vector<GLfloat> LIGHT_DIFFUSE{ 1.f, 1.f, 1.f, 1.f };  // Diffuse light
    const std::vector<GLfloat> LIGHT_POSITION{ 1.f, 1.f, 1.f, 0.f };  // Infinite light location
    const std::vector<GLfloat> COLOR_DIFFUSE{ 0.5f, 0.5f, 0.5f, 1.f };

    GLFWwindow* window_slave;
    GLuint fb, rbc, rbd, pbo;

    GLuint vao;
    GLuint vbuffer;
    GLuint listId;
    GLuint ibuffer;

    struct Vertex{
        GLfloat position[3];
        GLfloat normal[3];
        GLfloat texcoord[2];
    };

    struct Params{
        cv::Size cameraSize;
        float fl, cx, cy;
        float tx, ty, tz, rx, ry, rz;
    };

    Params params;
    Eigen::Matrix4f transformMatrix;

    Renderer(){
    }

    Eigen::Matrix4f getTransformMatrix(float roll, float pitch, float yaw, float x, float y, float z){
        Eigen::AngleAxisf rollAngle(roll / 180.0 * M_PI, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(pitch / 180.0 * M_PI, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle(yaw / 180.0 * M_PI, Eigen::Vector3f::UnitZ());
        Eigen::Quaternion<float> q = yawAngle * pitchAngle * rollAngle;

        Eigen::Matrix3f rotationMatrix = q.matrix();
        Eigen::Matrix4f transformMatrix = Eigen::Matrix4f::Identity();
        transformMatrix(0,0) = rotationMatrix(0,0);
        transformMatrix(0,1) = rotationMatrix(0,1);
        transformMatrix(0,2) = rotationMatrix(0,2);
        transformMatrix(0,3) = x;
        transformMatrix(1,0) = rotationMatrix(1,0);
        transformMatrix(1,1) = rotationMatrix(1,1);
        transformMatrix(1,2) = rotationMatrix(1,2);
        transformMatrix(1,3) = y;
        transformMatrix(2,0) = rotationMatrix(2,0);
        transformMatrix(2,1) = rotationMatrix(2,1);
        transformMatrix(2,2) = rotationMatrix(2,2);
        transformMatrix(2,3) = z;
        transformMatrix(3,0) = 0;
        transformMatrix(3,1) = 0;
        transformMatrix(3,2) = 0;
        transformMatrix(3,3) = 1;
        return transformMatrix;
    }

    cv::Point2i transformAndProject(Eigen::Vector3f point){
        Eigen::Vector3f transformedPoint = point;
        transformedPoint(0) = (point(0)*transformMatrix(0,0) + point(1)*transformMatrix(0,1) + point(2)*transformMatrix(0,2) + transformMatrix(0,3));
        transformedPoint(1) = (point(0)*transformMatrix(1,0) + point(1)*transformMatrix(1,1) + point(2)*transformMatrix(1,2) + transformMatrix(1,3));
        transformedPoint(2) = (point(0)*transformMatrix(2,0) + point(1)*transformMatrix(2,1) + point(2)*transformMatrix(2,2) + transformMatrix(2,3));
        cv::Point2i pixel;
        pixel.x =(int)(((params.fl*transformedPoint(0))/transformedPoint(2)) + params.cx);
        pixel.y =(int)(((params.fl*transformedPoint(1))/transformedPoint(2)) + params.cy);
        return pixel;
    }

    void setCameraParams(Params params){
        this->params = params;
        transformMatrix = getTransformMatrix(params.rx,params.ry,params.rz,params.tx,params.ty,params.tz);
        if(params.cameraSize.width != renderWidth || params.cameraSize.height != renderHeight){
            renderWidth = params.cameraSize.width;
            renderHeight = params.cameraSize.height;
        }
    }

    void reshapeRenderer(){
        glfwSetWindowSize(window_slave, renderWidth, renderHeight);
    }

    void startOnThread(std::string name){
        // OpenGL - Initialization
        std::cout << "Initializing.." << std::endl;
        //glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
        window_slave = glfwCreateWindow(renderWidth, renderHeight, name.c_str(), 0, 0);
        glfwMakeContextCurrent(window_slave);
        glewInit();

        if(drawLighting){
            glLightfv(GL_LIGHT0, GL_AMBIENT, LIGHT_DIFFUSE.data());
            glLightfv(GL_LIGHT0, GL_DIFFUSE, LIGHT_DIFFUSE.data());
            glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION.data());
            glEnable(GL_LIGHT0);
            glEnable(GL_LIGHTING);
        }

        // Create and bind a VAO
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // Create and bind a BO for vertex data
        glGenBuffers(1, &vbuffer);
        glBindBuffer(GL_ARRAY_BUFFER, vbuffer);

        // Create and bind a BO for vertex data
        glGenBuffers(1, &ibuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuffer);

        glEnable(GL_LIGHTING);
        glEnable( GL_DEPTH_TEST );
        glShadeModel( GL_SMOOTH );
        glEnable( GL_CULL_FACE );
        glClearColor( 0, 0, 0, 0 );

        cout << "Initialized" << endl;
    }

    cv::Mat draw(Eigen::MatrixXf mV, Eigen::MatrixXf mF){
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glFlush();
        glEnable(GL_LIGHTING);
        glShadeModel( GL_SMOOTH );
        glEnable( GL_TEXTURE_2D );
        glEnable(GL_CULL_FACE);
        glEnable(GL_MULTISAMPLE);

        float width = params.cameraSize.width;
        float height = params.cameraSize.height;
        float fx = params.fl;
        float cx = params.cx;
        float fy = params.fl;
        float cy = params.cy;
        float xs = ((width/2)-cx);
        float ys = ((height/2)-cy);
        float fovy = (180.0 / M_PI) * (atan2(height/2, fy)) * 2;

        xs = xs;
        ys = ys;
        glViewport(xs, ys, (float)renderWidth/1, (float)renderHeight/1. );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        gluPerspective( fovy, (float)width/(float)height, 0.1, 100. );
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        std::vector<Vertex> vertexdata(mV.rows());
        for(int r=0; r<mV.rows(); r++){
            Vertex& v = vertexdata[r];
            for(int c=0; c<mV.cols(); c++){
                v.position[c] = mV(r,c);
            }
        }

        std::vector<GLushort> indexdata(mF.rows()*3);
        for(int r=0; r<mF.rows(); r++){
            indexdata[r*3 + 0] = mF(r,0);
            indexdata[r*3 + 1] = mF(r,1);
            indexdata[r*3 + 2] = mF(r,2);
            if(drawNormal){
                Vertex& v0 = vertexdata[mF(r,0)];
                Vertex& v1 = vertexdata[mF(r,1)];
                Vertex& v2 = vertexdata[mF(r,2)];
                float x = (v1.position[1]-v0.position[1])*(v2.position[2]-v0.position[2])-(v1.position[2]-v0.position[2])*(v2.position[1]-v0.position[1]);
                float y = (v1.position[2]-v0.position[2])*(v2.position[0]-v0.position[0])-(v1.position[0]-v0.position[0])*(v2.position[2]-v0.position[2]);
                float z = (v1.position[0]-v0.position[0])*(v2.position[1]-v0.position[1])-(v1.position[1]-v0.position[1])*(v2.position[0]-v0.position[0]);
                float length = std::sqrt( x*x + y*y + z*z );
                x /= length;
                y /= length;
                z /= length;
                for(int i=0; i<3; i++){
                    vertexdata[mF(r,i)].normal[0] = x;
                    vertexdata[mF(r,i)].normal[1] = y;
                    vertexdata[mF(r,i)].normal[2] = z;
                }
            }
        }

        glEnable( GL_TEXTURE_2D );
        glEnable( GL_NORMALIZE );
        glColor4f( 0.0f, 0.0f, 0.0f, 0.0f );

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbuffer);

        // copy data into the buffer object
        glBufferData(GL_ARRAY_BUFFER, vertexdata.size() * sizeof(Vertex), &vertexdata[0], GL_STATIC_DRAW);

        // set up vertex attributes
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, sizeof(Vertex), (void*)offsetof(Vertex, position)); // vertices
        glEnableClientState(GL_NORMAL_ARRAY);
        glNormalPointer(GL_FLOAT, sizeof(Vertex), (void*)offsetof(Vertex, normal)); // normals
        glClientActiveTexture(GL_TEXTURE0);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glTexCoordPointer(2, GL_FLOAT, sizeof(Vertex), (void*)offsetof(Vertex, texcoord)); // normal

        // Create and bind a BO for index data
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuffer);

        // copy data into the buffer object
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexdata.size() * sizeof(GLushort), &indexdata[0], GL_STATIC_DRAW);

        // Set Transforms
        glPushMatrix();
        glTranslatef(-params.tx,-params.ty,-params.tz);
        glRotatef(params.rz, 0.0, 0.0, 1.0);
        glRotatef(params.ry, 0.0, 1.0, 0.0);
        glRotatef(params.rx+180, 1.0, 0.0, 0.0);

        // Draw
        if(drawContour){
            glDisable(GL_LIGHTING);
            glPolygonMode(GL_BACK, GL_LINE);
            glLineWidth(1);
            glCullFace(GL_FRONT);
            glColor3f(1., 1., 1.);
            glBindVertexArray(0);
            glBindVertexArray(vao);
            glDrawElements(GL_TRIANGLES, indexdata.size(), GL_UNSIGNED_SHORT, (void*)0);
            glCullFace(GL_BACK);
            glPolygonMode(GL_BACK, GL_FILL);
            glEnable(GL_LIGHTING);
            glBindVertexArray(0);
            glBindVertexArray(vao);
            glDrawElements(GL_TRIANGLES, indexdata.size(), GL_UNSIGNED_SHORT, (void*)0);
        }else{
            glBindVertexArray(0);
            glBindVertexArray(vao);
            glDrawElements(GL_TRIANGLES, indexdata.size(), GL_UNSIGNED_SHORT, (void*)0);
        }

        glPopMatrix();
        glDisable( GL_NORMALIZE );
        glDisable( GL_TEXTURE_2D );

        std::vector<std::uint8_t> data(renderWidth*renderHeight*4);
        glReadBuffer(GL_BACK);
        glReadPixels(0,0,renderWidth,renderHeight,GL_BGRA,GL_UNSIGNED_BYTE,&data[0]);
        cv::Mat m(cv::Size(renderWidth, renderHeight),CV_8UC4, &data[0]);
        //cv::flip(m, m, -1);
        cv::flip(m, m, 1);
        cv::flip(m, m, -1);
        cv::Rect rect(renderWidth/2 - width/2, renderHeight/2 - height/2, width, height);
        cv::Mat out = m(rect);

        //            // My Test
        //            for(auto v : vertexdata){
        //                Eigen::Vector3f point(v.position[0],v.position[1],v.position[2]);
        //                cv::Point2i pixel = transformAndProject(point);
        //                cv::circle(out, pixel, 2, cv::Scalar(255,0,0,255));
        //            }

        glfwSwapBuffers(window_slave);
        return out.clone();
    }

};

class RendererManager{

private:
    std::vector<std::thread> rendererThreads;
    std::vector<bool> renderThreadCompletes;
    std::mutex sendMtx, recvMtx, writebackLock;
    std::condition_variable sendCv;
    std::chrono::steady_clock::time_point begin, end;

public:
    typedef std::pair<Eigen::MatrixXf*,Eigen::MatrixXf*> RenderData;
    std::vector<RenderData> renderDatas;
    std::vector<Renderer::Params> renderParams;
    std::vector<cv::Mat> renderOutputs;
    std::vector<std::vector<std::vector<cv::Point>>> renderContours;
    std::vector<std::vector<cv::Point>> renderPoints;

    static void thread_worker(int id, RendererManager* manager){
        Renderer r;
        r.drawNormal = false;
        r.drawContour = true;
        r.setCameraParams(manager->renderParams[id]);
        r.startOnThread(std::to_string(id));

        while(1){
            std::unique_lock<std::mutex> lck(manager->sendMtx);
            manager->sendCv.wait(lck);
            lck.unlock();

            // Work
            r.setCameraParams(manager->renderParams[id]);
            cv::Mat out = r.draw(*manager->renderDatas[id].first,*manager->renderDatas[id].second);
            cv::cvtColor(out, out, CV_BGRA2GRAY);
            std::vector<std::vector<cv::Point>> contours;
            cv::Mat clone = out.clone();
            findContoursCV(clone, contours);

            // Convert
            cv::threshold(out, out, 250, 255, cv::THRESH_BINARY);
            std::vector<cv::Point> points;
            cv::findNonZero(out, points);

            manager->writebackLock.lock();
            manager->renderPoints[id] = points;
            manager->renderOutputs[id] = out;
            manager->renderContours[id] = contours;
            manager->renderThreadCompletes[id] = true;
            bool done = (std::find(std::begin(manager->renderThreadCompletes), std::end(manager->renderThreadCompletes), false) == std::end(manager->renderThreadCompletes));
            manager->writebackLock.unlock();
            if(done)
                manager->recvMtx.unlock();
        }
    }

    RendererManager(){

    }

    void addThread(Renderer::Params params){
        renderPoints.push_back(std::vector<cv::Point>());
        renderContours.push_back(std::vector<std::vector<cv::Point>>());
        renderDatas.push_back(RenderData());
        renderOutputs.push_back(cv::Mat());
        renderParams.push_back(params);
        renderThreadCompletes.push_back(false);
        rendererThreads.push_back(std::thread(RendererManager::thread_worker, rendererThreads.size(), this));
    }

    void signal(){
        writebackLock.lock();
        for(auto i : renderThreadCompletes) i = 0;
        writebackLock.unlock();
        recvMtx.lock();
        std::unique_lock<std::mutex> sendLck(sendMtx);
        sendCv.notify_all();
        sendLck.unlock();
    }

    void wait(){
        recvMtx.lock();
        recvMtx.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    void join(){
        for (int i = 0; i < rendererThreads.size(); ++i) {
            rendererThreads[i].join();
        }
    }

    static void findContoursCV(cv::Mat& img, cv::OutputArrayOfArrays _contours){
        IplImage* iplImg = new IplImage(img);
        CvMemStorage *storage = cvCreateMemStorage(0);
        CvSeq *_ccontours = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
        cvFindContours(iplImg, storage, &_ccontours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        delete iplImg;

        if( !_ccontours )
        {
            _contours.clear();
            return;
        }
        cv::Seq<CvSeq*> all_contours(cvTreeToNodeSeq( _ccontours, sizeof(CvSeq), storage ));
        int i, total = (int)all_contours.size();
        _contours.create(total, 1, 0, -1, true);
        cv::SeqIterator<CvSeq*> it = all_contours.begin();
        for( i = 0; i < total; i++, ++it )
        {
            CvSeq* c = *it;
            ((CvContour*)c)->color = (int)i;
            _contours.create((int)c->total, 1, CV_32SC2, i, true);
            cv::Mat ci = _contours.getMat(i);
            CV_Assert( ci.isContinuous() );
            cvCvtSeqToArray(c, ci.ptr());
        }

        return;
    }

};



#endif // OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP
