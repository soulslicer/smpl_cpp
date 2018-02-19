//#include <mutex>
//#include <stdio.h>
//#include <GL/glew.h>

//#include <GL/gl.h>
//#include <GL/glut.h>
//#include <GL/freeglut_ext.h>
//#include <GL/freeglut_std.h>
//#include <GL/glu.h>
//#include <opencv2/opencv.hpp>
//#include <GLFW/glfw3.h>
//#include <chrono>
//#include <thread>
//#include <fstream>
//#include <mutex>
//#include <functional>
//#include <eigen3/Eigen/Eigen>
//#include <iostream>
//using namespace std;
//#define BUFFER_OFFSET(i) ((char *)NULL + (i))

//#include <GL/glew.h>
//#include <GLFW/glfw3.h>
//#include <vector>
//#include <cmath>
//#include <cstdio>
//#include <limits>
//#include <chrono>
//#include <thread>
//#include <mutex>

////#define GL33
////#define FULLSCREEN


//static const int width = 640;
//static const int height = 360;

//GLFWwindow* window;
//GLFWwindow* window_slave;
//GLuint fb[2] = {std::numeric_limits<GLuint>::max(), std::numeric_limits<GLuint>::max()}; //framebuffers
//GLuint rb[2] = {std::numeric_limits<GLuint>::max(), std::numeric_limits<GLuint>::max()}; //renderbuffers, color and depth

//bool threadShouldRun = true;
//bool isFBOdirty = true; //true when last frame was displayed, false
//                        //when just updated and not yet displayed
//bool isFBOready = false; //set by worker thread when initialized
//bool isFBOsetupOnce = false; //set by main thread when initialized

//std::timed_mutex mutexGL;

//static bool checkFrameBuffer(GLuint fbuffer)
//{
//    bool isFB = glIsFramebuffer(fbuffer);
//    bool isCA = glIsRenderbuffer(rb[0]);
//    bool isDSA = glIsRenderbuffer(rb[1]);
//    bool isComplete = false;
//    if(isFB){
//        glBindFramebuffer(GL_FRAMEBUFFER, fbuffer);
//        isComplete = (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
//    }

///*
//    printf("Is fb a framebuffer? %s\n", isFB ? "[yes]" : "[no]");
//    printf("Is rb[0] a color renderbuffer? %s\n", isCA ? "[yes]" : "[no]");
//    printf("Is rv[1] a depth stencil renderbuffer? %s\n", isDSA ? "[yes]" : "[no]");
//    printf("Is fb framebuffer-complete? %s\n", isComplete ? "[yes]" : "[no]");
//*/
//    return isFB && isCA &&isDSA && isComplete;
//}

///* worker thread creates its own FBO to render to, as well as two renderbuffers.
//The renderbuffers are also used by a separate FBO in main()
//fb[0] is owned by main, and fb[1] is owned by worker thread
//*/
//static bool createFrameBuffer() //for worker thread
//{
//    bool ret;

//    glGenFramebuffers(1, &fb[1]);
//    glBindFramebuffer(GL_FRAMEBUFFER, fb[1]);
//    glGenRenderbuffers(2, rb);
//    glBindRenderbuffer(GL_RENDERBUFFER, rb[0]);
//    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 2, GL_RGBA8, width, height);
//    glBindRenderbuffer(GL_RENDERBUFFER, rb[1]);
//    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 2, GL_DEPTH24_STENCIL8, width, height);

//    glBindRenderbuffer(GL_RENDERBUFFER, rb[0]);
//    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb[0]);
//    glBindRenderbuffer(GL_RENDERBUFFER, rb[1]);
//    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rb[1]);

//    glFlush();

//    if(!(ret = checkFrameBuffer(fb[1]))){
//        glDeleteRenderbuffers(2, rb);
//        glDeleteFramebuffers(1,  &fb[1]);
//        glBindFramebuffer(GL_FRAMEBUFFER, 0);
//    }
//    return ret;
//}

///* If worker thread is finished initializing renderbuffers, reuse them in a FBO for main.
//fb[0] is owned by main, and fb[1] is owned by worker thread */
//static void createFrameBufferMain()
//{
//    while(!mutexGL.try_lock_for(std::chrono::seconds(1))){
//        return;
//    }

//    if(isFBOready){ //is other thread finished setting up FBO?
//        if(glIsRenderbuffer(rb[0]) && glIsRenderbuffer(rb[1])){
//            glBindFramebuffer(GL_FRAMEBUFFER, fb[0]);
//            glBindRenderbuffer(GL_RENDERBUFFER, rb[0]);
//            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb[0]);
//            glBindRenderbuffer(GL_RENDERBUFFER, rb[1]);
//            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rb[1]);
//            isFBOsetupOnce = true;
//        }
//    }
//    mutexGL.unlock();
//}

///* Used in main to copy from fbo into the real framebuffer.
//fb[0] FBO owned by main reuses renderbuffers from worker thread.
//The contents of those renderbuffers are then copied into the
//default framebuffer by this function. */
//static void copyFrameBuffer(GLuint fbuffer)
//{
//    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbuffer);
//    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
//    glBlitFramebuffer(0, 0, width, height,
//                      0, 0, width, height,
//                      GL_COLOR_BUFFER_BIT,
//                      GL_NEAREST);
//    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
//    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
//}

//static GLFWwindow* initWindow(GLFWwindow* shared, bool visible)
//{
//    GLFWwindow* win;
//#ifdef GL33
//    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//#endif
//    if(visible)
//        glfwWindowHint(GLFW_VISIBLE, GL_TRUE);
//    else
//        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

//#ifdef FULLSCREEN
//    GLFWmonitor* monitor = 0;
//    if(visible) //Don't create fullscreen window for offscreen contexts
//        monitor = glfwGetPrimaryMonitor();
//    win = glfwCreateWindow(width, height, "Optimus example", monitor, shared);
//#else
//    win = glfwCreateWindow(width, height, "Optimus example", 0, shared);
//#endif
//    return win;
//}

//static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
//{
//    if ((key == GLFW_KEY_ESCAPE || key == GLFW_KEY_ENTER)
//        && action == GLFW_PRESS){
//        glfwSetWindowShouldClose(window, GL_TRUE);
//    }
//}

///********************************************** TEST *********************************/
//static void worker_thread()
//{
//    glfwMakeContextCurrent(window_slave);
//    //create new shared framebuffer object
//    mutexGL.lock();
//    createFrameBuffer();
//    isFBOready = true;
//    mutexGL.unlock();

//    for(;;){
//        mutexGL.lock();
//        if(!threadShouldRun){
//            mutexGL.unlock();
//            break;
//        }
//        if(isFBOdirty){
//            glBindFramebuffer(GL_FRAMEBUFFER, fb[1]);
//            float r = (float)rand() / (float)RAND_MAX;
//            glClearColor(r,r,r,1.0f);
//            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//            // Draw?
//            glMatrixMode( GL_MODELVIEW );
//            glLoadIdentity();
//            glEnable(GL_LIGHTING);
//            glShadeModel( GL_SMOOTH );
//            glEnable( GL_TEXTURE_2D );
//            glPushMatrix();
//            glTranslatef(0,-0.5,-0.5);
//            glBegin(GL_POLYGON);
//            glVertex3f(0,0,-3);
//            glVertex3f(-1,-1,-3);
//            glVertex3f(1,-1,-3);
//            glEnd();
//            glPopMatrix();

//            glFlush();


//            isFBOdirty = false;
//        }
//        mutexGL.unlock();
//    }
//    printf("Exiting thread..\n");
//    return;
//}

//int main(int argc, char* argv[])
//{
//    if(!glfwInit()){
//        printf("Failed to initialize glfw\n");
//        return 0;
//    }
//    //main window
//    window = initWindow(0, true);
//    //window used by second thread
//    window_slave = initWindow(window, false);

//    if(!window || !window_slave){
//        glfwTerminate();
//        printf("Failed to create glfw windows\n");
//        return 0;
//    }
//    glfwSetKeyCallback(window, key_callback);
//    glfwMakeContextCurrent(window);

//    if(glewInit()){
//        printf("Failed to init GLEW\n");
//        glfwDestroyWindow(window);
//        glfwDestroyWindow(window_slave);
//        glfwTerminate();
//        return 0;
//    }

//    std::thread gl_thread(worker_thread);

//    glGenFramebuffers(1, &fb[0]);
//    glViewport(0, 0, width, height);

//    while(!glfwWindowShouldClose(window)){
//        glfwPollEvents(); //get key input
//        if(!isFBOsetupOnce){
//            createFrameBufferMain(); //isFBOsetupOnce = true when FBO can be used
//        } else {
//            if(checkFrameBuffer(fb[0])){
//                if(!mutexGL.try_lock_for(std::chrono::seconds(1)))
//                    continue;
//                if(!isFBOdirty){
//                    copyFrameBuffer(fb[0]);
//                    glfwSwapBuffers(window);
//                    isFBOdirty = true;
//                    printf("Framebuffer OK\n");
//                } else {
//                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
//                    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
//                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//                    printf("Framebuffer dirty\n");
//                }
//                mutexGL.unlock();
//            } else {
//                printf("Framebuffer not ready!\n");
//                GLenum e = glGetError();
//                printf("OpenGL error: %X\n", e);
//            }
//        }
//    }
//    threadShouldRun = false; //other thread only reads this
//    gl_thread.join();
//    glfwDestroyWindow(window);
//    glfwDestroyWindow(window_slave);
//    glfwTerminate();
//    return 0;
//}

////class ThreadManager{
////public:
////    std::vector<std::thread> threads;
////    std::vector<bool> threadCompletes;

////    std::mutex sendMtx, boolLock, recvMtx;
////    std::condition_variable sendCv;

////    static std::string printBool(std::vector<bool>& bools){
////        std::string xx;
////        for(int i=0; i<bools.size(); i++){
////            if(bools[i]) xx += "1"; else xx+="0";
////        }
////        return xx;
////    }

////    static void thread_worker(int id, ThreadManager* manager){

////        std::mutex internal;

////        while(1){
////            // Wait for main thread signal
////            std::unique_lock<std::mutex> lck(manager->sendMtx);
////            manager->boolLock.lock(); cout << "Thread Wait: " << id << endl; manager->boolLock.unlock();
////            manager->sendCv.wait(lck);
////            lck.unlock();

////            manager->boolLock.lock(); cout << "Start ParOp..: " << id << endl; manager->boolLock.unlock();
////            std::this_thread::sleep_for(std::chrono::milliseconds(5));

////            manager->boolLock.lock();
////            manager->threadCompletes[id] = true;
////            bool done = (std::find(std::begin(manager->threadCompletes), std::end(manager->threadCompletes), false) == std::end(manager->threadCompletes));
////            cout << "Job Complete: " << id << " " << printBool(manager->threadCompletes) << endl;
////            manager->boolLock.unlock();

////            if(done){
////                cout << "All Done" << endl;
////                done = false;
////                //manager->sendCv.notify_one();
////                manager->recvMtx.unlock();
////            }
////        }
////    }

////    ThreadManager(){

////    }

////    void addThread(){
////        threadCompletes.push_back(false);
////        threads.push_back(std::thread(ThreadManager::thread_worker, threads.size(), this));
////    }

////    void signal(){
////        boolLock.lock();
////        for(auto i : threadCompletes) i = 0;
////        boolLock.unlock();
////        std::unique_lock<std::mutex> sendLck(sendMtx);
////        //anotherMutex.lock();
////        cout << "Notifying.." << endl;
////        sendCv.notify_all();
////        sendLck.unlock();
////    }

////    void wait(){
////        recvMtx.lock();
////        recvMtx.unlock();
////        //std::unique_lock<std::mutex> sendLck(sendMtx);
////        //sendCv.wait(sendLck);

////        std::this_thread::sleep_for(std::chrono::milliseconds(1));
////        cout << "Ready" << endl;

////    }

////    void join(){
////        for (int i = 0; i < threads.size(); ++i) {
////            threads[i].join();
////        }
////    }
////};


////void test2(){

////    ThreadManager manager;
////    for(int i=0; i<4; i++){
////        manager.addThread();
////    }

////    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

////    for(int i=0; i<100; i++){
////        manager.signal();
////        manager.wait();
////        cout << "\nComplete: " << i << endl << endl;
////        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
////    }

////    cout << "ALL DONE" << endl;

////    manager.join();

////}

//#include <stdio.h>
//#include <stdlib.h>
//#include <GL/glut.h>

//void display(void);
//void idle(void);
//void draw_object(void);
//void reshape(int x, int y);
//void keyboard(unsigned char key, int x, int y);

//int outline = 1;

//int main(int argc, char **argv)
//{
//    glutInit(&argc, argv);
//    glutInitWindowSize(800, 600);
//    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE);

//    glutCreateWindow("object outline example");

//    glutDisplayFunc(display);
//    glutIdleFunc(idle);
//    glutReshapeFunc(reshape);
//    glutKeyboardFunc(keyboard);

//    glEnable(GL_LIGHTING);
//    glEnable(GL_LIGHT0);
//    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_CULL_FACE);
//    glEnable(GL_MULTISAMPLE);

//    glutMainLoop();
//    return 0;
//}


//void display(void)
//{
//    unsigned int msec = glutGet(GLUT_ELAPSED_TIME);
//    float t = (float)msec / 50.0f;

//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    glMatrixMode(GL_MODELVIEW);
//    glLoadIdentity();
//    glTranslatef(0, 0, -5);
//    glRotatef(t, 1, 0, 0);
//    glRotatef(t, 0, 1, 0);

//    if(outline) {
//        glDisable(GL_LIGHTING);
//        glPolygonMode(GL_BACK, GL_LINE);
//        glLineWidth(3);
//        glCullFace(GL_FRONT);
//        glColor3f(1, 0.5, 0.2);
//        draw_object();
//        glCullFace(GL_BACK);
//        glPolygonMode(GL_BACK, GL_FILL);
//        glEnable(GL_LIGHTING);
//    }
//    draw_object();

//    glutSwapBuffers();
//}

//void idle(void)
//{
//    glutPostRedisplay();
//}

//void draw_object(void)
//{
//    glutSolidTorus(0.5, 1.0, 16, 24);
//}

//void reshape(int x, int y)
//{
//    glViewport(0, 0, x, y);
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//    gluPerspective(50, (float)x / (float)y, 0.5, 500.0);
//}

//void keyboard(unsigned char key, int x, int y)
//{
//    switch(key) {
//    case 27:
//        exit(0);

//    case ' ':
//        outline = !outline;
//        break;
//    }
//}


#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <pf.h>

class ParticleFilterExt : public ParticleFilter{
public:
    cv::Mat input;

    ParticleFilterExt(int particleCount, int paramsCount) : ParticleFilter(particleCount, paramsCount)
    {
    }

    void weightFunction(){
        ParticleFilter::Probability intensityMeanProb(50);
        for(int i=0; i<particleCount_; i++){
            cv::Point pixel(stateMatrix_(0,i),stateMatrix_(1,i));
            cv::Vec3b colour = input.at<cv::Vec3b>(pixel);
            weightVector_(i,0) = intensityMeanProb.getProbability(255-colour[0]).log;
        }
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

};

int main(){
    ParticleFilterExt pf(600,2);
    Eigen::MatrixXf noiseVector(2,1);
    noiseVector << 2,2;
    Eigen::MatrixXf initialVal(2,2);
    initialVal <<   320,1,
                    240,1;
    Eigen::MatrixXf rangeMatrix(2,2);
    rangeMatrix <<  0,640,
                    0,480;
    pf.setRange(rangeMatrix);
    pf.setNoise(noiseVector);
    pf.initGauss(initialVal);



    int pos = 200;

    while(1){
        cv::Mat img(cv::Size(640,480), CV_8UC3, cv::Scalar(0,0,0));
        cv::circle(img, cv::Point(pos++,200), 70, cv::Scalar(255,255,255),CV_FILLED);
        cv::Mat mSource_Gray,mDist,mBlobDist;
        cv::cvtColor(img, mSource_Gray, CV_BGR2GRAY);
        cv::distanceTransform(mSource_Gray, mDist, CV_DIST_L2, 3);
        cv::normalize(mDist, mDist, 0, 1., cv::NORM_MINMAX);
        mDist.convertTo(mDist,CV_8UC1,255,0);
        cv::cvtColor(mDist,img,CV_GRAY2BGR);

        pf.input = img.clone();

        pf.update();
        pf.weightFunction();
        pf.resampleParticles();
        cv::Mat out = pf.draw();
        cv::imshow("win",out);
        cv::waitKey(15);
    }

    return 0;
}























