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

const int renderWidth = 640;
const int renderHeight = 480;

class Renderer{
public:
    GLFWwindow* window_slave;
    GLuint fb, rbc, rbd, pbo;

    struct Vertex{
      GLfloat position[3];
      GLfloat normal[3];
      GLfloat texcoord[2];
    };

    bool createFrameBuffers() //for worker thread
    {
        bool ret;

        glGenFramebuffers(1, &fb);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glGenRenderbuffers(1, &rbc);
        glBindRenderbuffer(GL_RENDERBUFFER, rbc);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 2, GL_RGBA8, renderWidth, renderHeight);
        glGenRenderbuffers(1, &rbd);
        glBindRenderbuffer(GL_RENDERBUFFER, rbd);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 2, GL_DEPTH24_STENCIL8, renderWidth, renderHeight);

        glBindRenderbuffer(GL_RENDERBUFFER, rbc);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbc);
        glBindRenderbuffer(GL_RENDERBUFFER, rbd);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbd);

        glGenBuffers(1,&pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, renderWidth*renderHeight*4, NULL, GL_DYNAMIC_READ);

        glFlush();

        return ret;
    }

    Renderer(){

    }

    void startOnThread(std::string name){
        // OpenGL - Initialization
        std::cout << "Initializing.." << std::endl;
        char *myargv [1];
        int myargc=1;
        if(name == "0"){
            myargv[0]=strdup ("GLUT");
            glutInit(&myargc, myargv);
            glutInitDisplayMode(GLUT_DOUBLE); // glEndList();Enable double buffered mode
            glutInitWindowSize(renderWidth, renderHeight);   // Set the window's initial width & height
            glutInitWindowPosition(50, 50); // Position the window's initial top-left corner
        }
        glutCreateWindow(name.c_str());          // Create window with the given title
        glutHideWindow();
        glewInit();
        glfwInit();

        //initWindow(window_slave,true);

        glEnable(GL_LIGHTING);
        glEnable( GL_DEPTH_TEST );
        glShadeModel( GL_SMOOTH );
        glEnable( GL_CULL_FACE );
        glClearColor( 1, 1, 1, 1 );

        //createFrameBuffers();
        cout << "ok" << endl;
    }

    cv::Mat draw(){
        glBindFramebuffer(GL_FRAMEBUFFER,fb);

        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glFlush();
        glEnable(GL_LIGHTING);
        glShadeModel( GL_SMOOTH );
        glEnable( GL_TEXTURE_2D );

        glViewport( 0, 0, (float)renderWidth/1, (float)renderHeight/1. );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        gluPerspective( 60, (float)renderWidth/(float)renderHeight, 0.1, 10000. );
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        // Draw?
        glPushMatrix();
        glTranslatef(0,-0.5,-0.5);
        glBegin(GL_POLYGON);
        glVertex3f(0,0,-3);
        glVertex3f(-1,-1,-3);
        glVertex3f(1,-1,-3);
        glEnd();
        glPopMatrix();

        // Read buffer test
        std::vector<std::uint8_t> data(renderWidth*renderHeight*4);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
        glReadBuffer(GL_BACK);
        glReadPixels(0,0,renderWidth,renderHeight,GL_BGRA,GL_UNSIGNED_BYTE,&data[0]);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo); //Might not be necessary...
        cv::Mat m(cv::Size(renderWidth, renderHeight),CV_8UC4, &data[0]);
        cv::flip(m, m, -1);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //glutSwapBuffers();
        //glutPostRedisplay();
        //glutMainLoopEvent();

        return m.clone();
    }

};

void thread_worker(const SMPL& smpl, int x){
    Renderer r;
    r.startOnThread(std::to_string(x));

    while(1){
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int test(){
//    Renderer r;
//    r.startOnThread();
//    cv::Mat a = r.draw();
//    while(1){
//        cv::imshow("win",a);
//        cv::waitKey(15);

//    }

    SMPL smpl;

    const int num_threads = 2;
    std::thread t[num_threads];

    //Launch a group of threads
    for (int i = 0; i < num_threads; ++i) {
        t[i] = std::thread(thread_worker, smpl, i);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "Launched from the main\n";

    //Join the threads with the main thread
    for (int i = 0; i < num_threads; ++i) {
        t[i].join();
    }
}

int main(int argc, char *argv[])
{
    //test();


    //testTensor();
    //exit(-1);
    bool active = true;
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
