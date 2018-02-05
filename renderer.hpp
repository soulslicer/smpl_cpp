#ifndef OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP
#define OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP

#include <mutex>
#include <stdio.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <GL/freeglut_std.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <fstream>
#include <mutex>
#include <functional>

namespace op
{
    struct OBJObject;
    struct OBJFace;
    struct OBJVertex;
    struct OBJNormal;
    struct OBJTexture;
    struct OBJFaceItem;
    struct OBJMaterial;

    class WObject
    {
    public:
        WObject();
        ~WObject();
        bool clearOBJFile();
        void print();
        bool loadOBJFile( const std::string& data_path, const std::string& mesh_filename, const std::string& material_filename );
        void render();
        void rebuild();

    private:
        std::string mDataPath;
        std::string mCurrentMaterial;
        std::shared_ptr<OBJObject> mObject;
        std::map<std::string,GLuint> textures;
        GLuint listId;
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
}

#endif // OPENPOSE_EXPERIMENTAL_3D_RENDERER_HPP
