#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <pf.h>

//void findContoursCV(cv::Mat& img, cv::OutputArrayOfArrays _contours){
//    IplImage* iplImg = new IplImage(img);
//    CvMemStorage *storage = cvCreateMemStorage(0);
//    CvSeq *_ccontours = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
//    cvFindContours(iplImg, storage, &_ccontours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
//    delete iplImg;

//    if( !_ccontours )
//    {
//        _contours.clear();
//        return;
//    }
//    cv::Seq<CvSeq*> all_contours(cvTreeToNodeSeq( _ccontours, sizeof(CvSeq), storage ));
//    int i, total = (int)all_contours.size();
//    _contours.create(total, 1, 0, -1, true);
//    cv::SeqIterator<CvSeq*> it = all_contours.begin();
//    for( i = 0; i < total; i++, ++it )
//    {
//        CvSeq* c = *it;
//        ((CvContour*)c)->color = (int)i;
//        _contours.create((int)c->total, 1, CV_32SC2, i, true);
//        cv::Mat ci = _contours.getMat(i);
//        CV_Assert( ci.isContinuous() );
//        cvCvtSeqToArray(c, ci.ptr());
//    }

//    return;
//}

//class Renderer{
//public:
//    int renderWidth = 640;
//    int renderHeight = 480;
//    bool drawNormal = false;
//    bool drawContour = true;
//    bool drawLighting = false;

//    const std::vector<GLfloat> LIGHT_DIFFUSE{ 1.f, 1.f, 1.f, 1.f };  // Diffuse light
//    const std::vector<GLfloat> LIGHT_POSITION{ 1.f, 1.f, 1.f, 0.f };  // Infinite light location
//    const std::vector<GLfloat> COLOR_DIFFUSE{ 0.5f, 0.5f, 0.5f, 1.f };

//    GLFWwindow* window_slave;
//    GLuint fb, rbc, rbd, pbo;

//    GLuint vao;
//    GLuint vbuffer;
//    GLuint listId;
//    GLuint ibuffer;

//    struct Vertex{
//        GLfloat position[3];
//        GLfloat normal[3];
//        GLfloat texcoord[2];
//    };

//    struct Params{
//        cv::Size cameraSize;
//        float fl, cx, cy;
//        float tx, ty, tz, rx, ry, rz;
//    };

//    Params params;
//    Eigen::Matrix4f transformMatrix;

//    Renderer(){
//    }

//    Eigen::Matrix4f getTransformMatrix(float roll, float pitch, float yaw, float x, float y, float z){
//        Eigen::AngleAxisf rollAngle(roll / 180.0 * M_PI, Eigen::Vector3f::UnitX());
//        Eigen::AngleAxisf pitchAngle(pitch / 180.0 * M_PI, Eigen::Vector3f::UnitY());
//        Eigen::AngleAxisf yawAngle(yaw / 180.0 * M_PI, Eigen::Vector3f::UnitZ());
//        Eigen::Quaternion<float> q = yawAngle * pitchAngle * rollAngle;

//        Eigen::Matrix3f rotationMatrix = q.matrix();
//        Eigen::Matrix4f transformMatrix = Eigen::Matrix4f::Identity();
//        transformMatrix(0,0) = rotationMatrix(0,0);
//        transformMatrix(0,1) = rotationMatrix(0,1);
//        transformMatrix(0,2) = rotationMatrix(0,2);
//        transformMatrix(0,3) = x;
//        transformMatrix(1,0) = rotationMatrix(1,0);
//        transformMatrix(1,1) = rotationMatrix(1,1);
//        transformMatrix(1,2) = rotationMatrix(1,2);
//        transformMatrix(1,3) = y;
//        transformMatrix(2,0) = rotationMatrix(2,0);
//        transformMatrix(2,1) = rotationMatrix(2,1);
//        transformMatrix(2,2) = rotationMatrix(2,2);
//        transformMatrix(2,3) = z;
//        transformMatrix(3,0) = 0;
//        transformMatrix(3,1) = 0;
//        transformMatrix(3,2) = 0;
//        transformMatrix(3,3) = 1;
//        return transformMatrix;
//    }

//    cv::Point2i transformAndProject(Eigen::Vector3f point){
//        Eigen::Vector3f transformedPoint = point;
//        transformedPoint(0) = (point(0)*transformMatrix(0,0) + point(1)*transformMatrix(0,1) + point(2)*transformMatrix(0,2) + transformMatrix(0,3));
//        transformedPoint(1) = (point(0)*transformMatrix(1,0) + point(1)*transformMatrix(1,1) + point(2)*transformMatrix(1,2) + transformMatrix(1,3));
//        transformedPoint(2) = (point(0)*transformMatrix(2,0) + point(1)*transformMatrix(2,1) + point(2)*transformMatrix(2,2) + transformMatrix(2,3));
//        cv::Point2i pixel;
//        pixel.x =(int)(((params.fl*transformedPoint(0))/transformedPoint(2)) + params.cx);
//        pixel.y =(int)(((params.fl*transformedPoint(1))/transformedPoint(2)) + params.cy);
//        return pixel;
//    }

//    void setCameraParams(Params params){
//        this->params = params;
//        transformMatrix = getTransformMatrix(params.rx,params.ry,params.rz,params.tx,params.ty,params.tz);
//        if(params.cameraSize.width != renderWidth || params.cameraSize.height != renderHeight){
//            renderWidth = params.cameraSize.width;
//            renderHeight = params.cameraSize.height;
//            glfwSetWindowSize(window_slave, renderWidth, renderHeight);
//        }
//    }

//    void startOnThread(std::string name){
//        // OpenGL - Initialization
//        std::cout << "Initializing.." << std::endl;
//        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
//        window_slave = glfwCreateWindow(renderWidth, renderHeight, name.c_str(), 0, 0);
//        glfwMakeContextCurrent(window_slave);
//        glewInit();

//        if(drawLighting){
//            glLightfv(GL_LIGHT0, GL_AMBIENT, LIGHT_DIFFUSE.data());
//            glLightfv(GL_LIGHT0, GL_DIFFUSE, LIGHT_DIFFUSE.data());
//            glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION.data());
//            glEnable(GL_LIGHT0);
//            glEnable(GL_LIGHTING);
//        }

//        // Create and bind a VAO
//        glGenVertexArrays(1, &vao);
//        glBindVertexArray(vao);

//        // Create and bind a BO for vertex data
//        glGenBuffers(1, &vbuffer);
//        glBindBuffer(GL_ARRAY_BUFFER, vbuffer);

//        // Create and bind a BO for vertex data
//        glGenBuffers(1, &ibuffer);
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuffer);

//        glEnable(GL_LIGHTING);
//        glEnable( GL_DEPTH_TEST );
//        glShadeModel( GL_SMOOTH );
//        glEnable( GL_CULL_FACE );
//        glClearColor( 0, 0, 0, 0 );

//        cout << "Initialized" << endl;
//    }

//    cv::Mat draw(Eigen::MatrixXf mV, Eigen::MatrixXf mF){
//        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//        glFlush();
//        glEnable(GL_LIGHTING);
//        glShadeModel( GL_SMOOTH );
//        glEnable( GL_TEXTURE_2D );
//        glEnable(GL_CULL_FACE);
//        glEnable(GL_MULTISAMPLE);

//        float width = params.cameraSize.width;
//        float height = params.cameraSize.height;
//        float fx = params.fl;
//        float cx = params.cx;
//        float fy = params.fl;
//        float cy = params.cy;
//        float xs = ((width/2)-cx);
//        float ys = ((height/2)-cy);
//        float fovy = (180.0 / M_PI) * (atan2(height/2, fy)) * 2;

//        xs = xs;
//        ys = ys;
//        glViewport(xs, ys, (float)renderWidth/1, (float)renderHeight/1. );
//        glMatrixMode( GL_PROJECTION );
//        glLoadIdentity();
//        gluPerspective( fovy, (float)width/(float)height, 0.1, 100. );
//        glMatrixMode( GL_MODELVIEW );
//        glLoadIdentity();

//        std::vector<Vertex> vertexdata(mV.rows());
//        for(int r=0; r<mV.rows(); r++){
//            Vertex& v = vertexdata[r];
//            for(int c=0; c<mV.cols(); c++){
//                v.position[c] = mV(r,c);
//            }
//        }

//        std::vector<GLushort> indexdata(mF.rows()*3);
//        for(int r=0; r<mF.rows(); r++){
//            indexdata[r*3 + 0] = mF(r,0);
//            indexdata[r*3 + 1] = mF(r,1);
//            indexdata[r*3 + 2] = mF(r,2);
//            if(drawNormal){
//                Vertex& v0 = vertexdata[mF(r,0)];
//                Vertex& v1 = vertexdata[mF(r,1)];
//                Vertex& v2 = vertexdata[mF(r,2)];
//                float x = (v1.position[1]-v0.position[1])*(v2.position[2]-v0.position[2])-(v1.position[2]-v0.position[2])*(v2.position[1]-v0.position[1]);
//                float y = (v1.position[2]-v0.position[2])*(v2.position[0]-v0.position[0])-(v1.position[0]-v0.position[0])*(v2.position[2]-v0.position[2]);
//                float z = (v1.position[0]-v0.position[0])*(v2.position[1]-v0.position[1])-(v1.position[1]-v0.position[1])*(v2.position[0]-v0.position[0]);
//                float length = std::sqrt( x*x + y*y + z*z );
//                x /= length;
//                y /= length;
//                z /= length;
//                for(int i=0; i<3; i++){
//                    vertexdata[mF(r,i)].normal[0] = x;
//                    vertexdata[mF(r,i)].normal[1] = y;
//                    vertexdata[mF(r,i)].normal[2] = z;
//                }
//            }
//        }

//        glEnable( GL_TEXTURE_2D );
//        glEnable( GL_NORMALIZE );
//        glColor4f( 0.0f, 0.0f, 0.0f, 0.0f );

//        glBindVertexArray(vao);
//        glBindBuffer(GL_ARRAY_BUFFER, vbuffer);

//        // copy data into the buffer object
//        glBufferData(GL_ARRAY_BUFFER, vertexdata.size() * sizeof(Vertex), &vertexdata[0], GL_STATIC_DRAW);

//        // set up vertex attributes
//        glEnableClientState(GL_VERTEX_ARRAY);
//        glVertexPointer(3, GL_FLOAT, sizeof(Vertex), (void*)offsetof(Vertex, position)); // vertices
//        glEnableClientState(GL_NORMAL_ARRAY);
//        glNormalPointer(GL_FLOAT, sizeof(Vertex), (void*)offsetof(Vertex, normal)); // normals
//        glClientActiveTexture(GL_TEXTURE0);
//        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
//        glTexCoordPointer(2, GL_FLOAT, sizeof(Vertex), (void*)offsetof(Vertex, texcoord)); // normal

//        // Create and bind a BO for index data
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibuffer);

//        // copy data into the buffer object
//        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexdata.size() * sizeof(GLushort), &indexdata[0], GL_STATIC_DRAW);

//        // Set Transforms
//        glPushMatrix();
//        glTranslatef(-params.tx,-params.ty,-params.tz);
//        glRotatef(params.rz, 0.0, 0.0, 1.0);
//        glRotatef(params.ry, 0.0, 1.0, 0.0);
//        glRotatef(params.rx+180, 1.0, 0.0, 0.0);

//        // Draw
//        if(drawContour){
//            glDisable(GL_LIGHTING);
//            glPolygonMode(GL_BACK, GL_LINE);
//            glLineWidth(1);
//            glCullFace(GL_FRONT);
//            glColor3f(1., 1., 1.);
//            glBindVertexArray(0);
//            glBindVertexArray(vao);
//            glDrawElements(GL_TRIANGLES, indexdata.size(), GL_UNSIGNED_SHORT, (void*)0);
//            glCullFace(GL_BACK);
//            glPolygonMode(GL_BACK, GL_FILL);
//            glEnable(GL_LIGHTING);
//            glBindVertexArray(0);
//            glBindVertexArray(vao);
//            glDrawElements(GL_TRIANGLES, indexdata.size(), GL_UNSIGNED_SHORT, (void*)0);
//        }else{
//            glBindVertexArray(0);
//            glBindVertexArray(vao);
//            glDrawElements(GL_TRIANGLES, indexdata.size(), GL_UNSIGNED_SHORT, (void*)0);
//        }

//        glPopMatrix();
//        glDisable( GL_NORMALIZE );
//        glDisable( GL_TEXTURE_2D );

//        std::vector<std::uint8_t> data(renderWidth*renderHeight*4);
//        glReadBuffer(GL_BACK);
//        glReadPixels(0,0,renderWidth,renderHeight,GL_BGRA,GL_UNSIGNED_BYTE,&data[0]);
//        cv::Mat m(cv::Size(renderWidth, renderHeight),CV_8UC4, &data[0]);
//        cv::flip(m, m, -1);
//        cv::Rect rect(renderWidth/2 - width/2, renderHeight/2 - height/2, width, height);
//        cv::Mat out = m(rect);

//        // My Test
//        //for(auto v : vertexdata){
//        //    Eigen::Vector3f point(v.position[0],v.position[1],v.position[2]);
//        //    cv::Point2i pixel = transformAndProject(point);
//        //    cv::circle(out, pixel, 2, cv::Scalar(255,0,0));
//        //}

//        glfwSwapBuffers(window_slave);
//        return out.clone();
//    }

//};

//class RendererManager{

//private:
//    std::vector<std::thread> rendererThreads;
//    std::vector<bool> renderThreadCompletes;
//    std::mutex sendMtx, recvMtx, writebackLock;
//    std::condition_variable sendCv;
//    std::chrono::steady_clock::time_point begin, end;

//public:
//    typedef std::pair<Eigen::MatrixXf*,Eigen::MatrixXf*> RenderData;
//    std::vector<RenderData> renderDatas;
//    std::vector<Renderer::Params> renderParams;
//    std::vector<cv::Mat> renderOutputs;
//    std::vector<std::vector<cv::Point>> renderContours;

//    static void thread_worker(int id, RendererManager* manager){
//        Renderer r;
//        r.startOnThread(std::to_string(id));

//        while(1){
//            std::unique_lock<std::mutex> lck(manager->sendMtx);
//            manager->sendCv.wait(lck);
//            lck.unlock();

//            // Work
//            r.setCameraParams(manager->renderParams[id]);
//            cv::Mat out = r.draw(*manager->renderDatas[id].first,*manager->renderDatas[id].second);
//            cv::cvtColor(out, out, CV_BGRA2GRAY);
//            std::vector<std::vector<cv::Point>> contours;
//            findContoursCV(out, contours);

//            manager->writebackLock.lock();
//            manager->renderOutputs[id] = out;
//            manager->renderContours[id] = contours[0];
//            manager->renderThreadCompletes[id] = true;
//            bool done = (std::find(std::begin(manager->renderThreadCompletes), std::end(manager->renderThreadCompletes), false) == std::end(manager->renderThreadCompletes));
//            manager->writebackLock.unlock();
//            if(done)
//                manager->recvMtx.unlock();
//        }
//    }

//    RendererManager(){

//    }

//    void addThread(){
//        renderContours.push_back(std::vector<cv::Point>());
//        renderDatas.push_back(RenderData());
//        renderOutputs.push_back(cv::Mat());
//        renderParams.push_back(Renderer::Params());
//        renderThreadCompletes.push_back(false);
//        rendererThreads.push_back(std::thread(RendererManager::thread_worker, rendererThreads.size(), this));
//    }

//    void signal(){
//        writebackLock.lock();
//        for(auto i : renderThreadCompletes) i = 0;
//        writebackLock.unlock();
//        recvMtx.lock();
//        std::unique_lock<std::mutex> sendLck(sendMtx);
//        sendCv.notify_all();
//        sendLck.unlock();
//    }

//    void wait(){
//        recvMtx.lock();
//        recvMtx.unlock();
//        std::this_thread::sleep_for(std::chrono::milliseconds(1));
//    }

//    void join(){
//        for (int i = 0; i < rendererThreads.size(); ++i) {
//            rendererThreads[i].join();
//        }
//    }

//};

//int test(){
//    glfwInit();
//    //glewInit();

//    // Load SMPL
//    SMPL smpl;
//    smpl.loadModelFromJSONFile(std::string(CMAKE_CURRENT_SOURCE_DIR) + "/model.json");
//    std::cout.setstate(std::ios_base::failbit);
//    smpl.updateModel();
//    std::cout.clear();

//    // Params
//    int totalThreads = 10;
//    int totalOps = 1000;
//    std::chrono::steady_clock::time_point begin, end;
//    Renderer::Params params;
//    params.cameraSize = cv::Size(640,480);
//    params.fl = 500.;
//    params.cx = params.cameraSize.width/2-5;
//    params.cy = params.cameraSize.height/2+5;
//    params.tx = 0.2;
//    params.ty = 0.2;
//    params.tz = 2.5;
//    params.rx = 20;
//    params.ry = 20;
//    params.rz = 20;

////    // Single thread
////    Renderer r;
////    r.startOnThread("win");
////    r.setCameraParams(params);
////    cv::Mat out = r.draw(smpl.mVTemp2, smpl.mF);
////    begin = std::chrono::steady_clock::now();
////    for(int i=0; i<1; i++){
////        r.draw(smpl.mVTemp2, smpl.mF);
////    }
////    end= std::chrono::steady_clock::now();
////    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;

////    // Contour
////    begin = std::chrono::steady_clock::now();
////    cv::Mat convert;
////    cv::cvtColor(out, out, CV_BGRA2GRAY);
////    std::vector<std::vector<cv::Point>> contours;
////    findContoursCV(out, contours);
////    end= std::chrono::steady_clock::now();
////    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;
////    while(1){
////        cv::imshow("im", out);
////        cv::waitKey(15);
////    }

//    // Start Threads
//    RendererManager manager;
//    for(int i=0; i<totalThreads; i++){
//        std::this_thread::sleep_for(std::chrono::milliseconds(50));
//        manager.addThread();
//    }

//    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
//    begin = std::chrono::steady_clock::now();
//    for(int i=0; i<totalOps/totalThreads; i++){
//        // Set data
//        for(int i=0; i<totalThreads; i++){
//            manager.renderParams[i] = params;
//            manager.renderDatas[i] = RendererManager::RenderData(&smpl.mVTemp2, &smpl.mF);
//        }
//        manager.signal();
//        manager.wait();
//    }
//    cout << "**DONE**" << endl;
//    end= std::chrono::steady_clock::now();
//    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. << " ms" << std::endl;
//    manager.join();
//}


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
    Eigen::MatrixXf a(2,2);
    a(0,1) = 5;

    Eigen::MatrixXf b = a;
    b(0,0) = 6;
    cout << a << endl;
    cout << b << endl;
    exit(-1);

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























