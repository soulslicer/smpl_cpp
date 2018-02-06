#ifndef OBJECT_MESH_H
#define OBJECT_MESH_H

#include <time.h>
#include <eigen3/Eigen/Dense>
#include <GL/glut.h>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

using namespace std;

static const unsigned int renderWidth  = 640;
static const unsigned int renderHeight = 480;

inline static Eigen::Matrix4f getIdentity(){
        Eigen::Matrix4f i;
        i << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;
        return i;
}

class CameraObject
{
    cv::Mat cameraMatrix;

}

class ObjectMesh
{
private:
    struct OBJObject;
    struct OBJFace;
    struct OBJVertex;
    struct OBJNormal;
    struct OBJTexture;
    struct OBJFaceItem;
    struct OBJMaterial;

    struct OBJFace
    {
        std::vector<OBJFaceItem> items;
        std::string              material;
    };

    struct OBJFaceItem
    {
        OBJFaceItem()
        {
            vertexIndex  = 0;
            normalIndex  = 0;
            textureIndex = 0;
        }

        unsigned int vertexIndex;
        unsigned int normalIndex;
        unsigned int textureIndex;
    };

    struct OBJVertex
    {
        OBJVertex()
        {
            coords[0] = 0;
            coords[1] = 0;
            coords[2] = 0;
        }

        float coords[3];
    };

    struct OBJNormal
    {
        OBJNormal()
        {
            coords[0] = 0;
            coords[1] = 0;
            coords[2] = 0;
        }

        float coords[3];
    };

    struct OBJTexture
    {
        OBJTexture()
        {
            coords[0] = 0;
            coords[1] = 0;
            coords[2] = 0;
        }

        float coords[3];
    };

    struct OBJMaterial
    {
        float  Ka[4];
        float  Kd[4];
        float  Ks[4];

        GLuint texture;

        float texScaleU;
        float texScaleV;
    };

    struct OBJObject
    {
        OBJObject()
        {
            vertices.push_back( OBJVertex()  );
            normals.push_back ( OBJNormal()  );
            textures.push_back( OBJTexture() );
            computedNormals.push_back( OBJNormal() );
        }

        void clear(){
            vertices.clear();
            computedNormals.clear();
            normals.clear();
            textures.clear();
            faces.clear();
            materials.clear();
            vertices.push_back( OBJVertex()  );
            normals.push_back ( OBJNormal()  );
            textures.push_back( OBJTexture() );
            computedNormals.push_back( OBJNormal() );
        }

        std::vector<OBJVertex>            vertices;
        std::vector<OBJNormal>            computedNormals;
        std::vector<OBJNormal>            normals;
        std::vector<OBJTexture>           textures;
        std::vector<OBJFace>              faces;
        std::map<std::string,OBJMaterial> materials;
    };

    inline void chop( std::string& str )
    {
        std::string whitespaces (" \t\f\v\n\r");
        size_t found = str.find_last_not_of( whitespaces );
        if ( found!=std::string::npos )
            str.erase( found+1 );
        else
            str.clear();
    }

    inline void split( const std::string& str, char delim, std::vector<std::string>& tokens )
    {
        std::stringstream iss(str);
        std::string item;
        while ( std::getline(iss, item, delim) )
        {
            if ( item!=" " && !item.empty() )
            {
                chop(item);
                tokens.push_back(item);
            }
        }
    }

    std::string texType_ = ".png";
    std::string currentMaterial_;

private:
    OBJObject object_;
    std::map<std::string,GLuint> textures_;
    GLuint listId_;
    std::string dataPath_;
    bool glutInitialised_;

    Eigen::Matrix4f cloudPose_;
    Eigen::Vector3f cloudSize_;

    int loadMeshIntoGPU();
    void processMaterialLine( const std::string& line );
    void processMeshLine( const std::string& line );
    void build();
    bool load_texture( const std::string& filename, bool clamp );
    bool release_texture( const std::string& filename );
    GLuint get_texture( const std::string& filename );


public:
    ObjectMesh();
    ~ObjectMesh();

    bool loadOBJFile( const std::string& data_path, const std::string& mesh_filename="mesh.obj", const std::string& material_filename="mesh.mtl" );
    bool clearOBJFile();
    bool startRenderer();
    cv::Mat renderPerspective(DataCombiner::CameraObject& cameraObject, Eigen::Matrix4f cloudPose = getIdentity(), bool debug = false);
    cv::Mat renderOrtho(DataCombiner::CameraObject& cameraObject, float mult, float xl, float yl, Eigen::Matrix4f cloudPose = getIdentity(), bool debug = false);
    cv::Mat getMat();
    void print();

};

#endif

