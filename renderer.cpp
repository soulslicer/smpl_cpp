#include <renderer.hpp>

#define USE_COMPUTED_NORMALS

namespace op
{
const bool LOG_VERBOSE_3D_RENDERER = false;

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

    int resize(int count){
        vertices.resize(count+1);
        normals.resize(count+1);
        textures.resize(count+1);
        computedNormals.resize(count+1);
        faces.resize(count);
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

void log(std::string x){
    if(LOG_VERBOSE_3D_RENDERER)
    std::cout << x << std::endl;
}

void error(std::string x, const int line, const std::string& function, const std::string& file){
    std::cout << x << " " << line << " " << function << " " << file << std::endl;
}

static const unsigned int renderWidth  = 640;
static const unsigned int renderHeight = 480;
std::vector<std::shared_ptr<WObject>> wObjects;

void WObject::rebuild(int renderType)
{
    if (listId != 0)
        glDeleteLists( listId, 1 );

    glEnable( GL_TEXTURE_2D );
    listId = glGenLists(1);
    log(std::to_string(listId));
    glNewList( listId, GL_COMPILE );
    glColor4f( 1.0f, 1.0f, 1.0f, 1.0f );

    OBJObject& object = *mObject.get();
    glEnable( GL_NORMALIZE );
    glEnable( GL_TEXTURE_2D );

    static float ambient [4];
    static float diffuse [4];
    static float specular[4];

    // Should use vertex arrays somehow

    for ( int i=0; i<object.faces.size(); ++i ) // Each face
    {
        OBJFace& face = object.faces.at(i);
        OBJMaterial& material = object.materials[face.material];

        if(face.material.size()){
            glMaterialfv( GL_FRONT, GL_AMBIENT,  material.Ka );
            glMaterialfv( GL_FRONT, GL_DIFFUSE,  material.Kd );
            glMaterialfv( GL_FRONT, GL_SPECULAR, material.Ks );
            glBindTexture( GL_TEXTURE_2D, material.texture );
        }

        if(renderType == RENDER_NORMAL){
            glBegin( GL_POLYGON );
        }else if(renderType == RENDER_POINTS){
            glBegin( GL_POINTS );
        }else if(renderType == RENDER_WIREFRAME){
            glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
            glBegin( GL_POLYGON );
        }else{
            throw std::runtime_error("No such render type");
        }

        for ( int j=0; j<object.faces.at(i).items.size(); ++j ) // Each vertex
        {
            OBJFaceItem& item = object.faces.at(i).items.at(j);

#ifdef USE_COMPUTED_NORMALS
            glNormal3fv( object.computedNormals.at(item.vertexIndex).coords );
#else
            glNormal3fv( object.normals.at(item.vertexIndex).coords );
#endif

            if ( item.textureIndex > 0 ){
                glTexCoord2f( object.textures.at(item.textureIndex).coords[0]*material.texScaleU,
                        object.textures.at(item.textureIndex).coords[1]*material.texScaleV );
            }

            glVertex3fv( object.vertices.at(item.vertexIndex).coords );

        }
        glEnd();
    }

    if(renderType == RENDER_WIREFRAME){
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    }

    glDisable( GL_NORMALIZE );
    glDisable( GL_TEXTURE_2D );

    glEndList();
}

void WObject::render(){
    glCallList( listId );
}

bool WObject::clearOBJFile(bool clearObject){
    if (listId != 0)
        glDeleteLists( listId, 1 );
    if(clearObject)
        mObject->clear();

    std::vector<std::string> texKeys;
    for(std::map<std::string,GLuint>::iterator iterator = textures.begin(); iterator != textures.end(); iterator++) {
        texKeys.push_back(iterator->first);
    }
    for(int i=0; i<texKeys.size(); i++){
        releaseTexture(texKeys[i]);
    }
}

bool WObject::loadEigenData(const Eigen::MatrixXf& v, const Eigen::MatrixXf& f){
    if(v.rows() == mObject->vertices.size()-1)
        clearOBJFile(false);
    else
       clearOBJFile(true);

    OBJMaterial& defaultMaterial = mObject->materials[""];
    defaultMaterial.Ka[0] = 1;
    defaultMaterial.Ka[1] = 1;
    defaultMaterial.Ka[2] = 1;
    defaultMaterial.Ka[3] = 1;

    defaultMaterial.Kd[0] = 1;
    defaultMaterial.Kd[1] = 1;
    defaultMaterial.Kd[2] = 1;
    defaultMaterial.Kd[3] = 1;

    defaultMaterial.Ks[0] = 1;
    defaultMaterial.Ks[1] = 1;
    defaultMaterial.Ks[2] = 1;
    defaultMaterial.Ks[3] = 1;

    defaultMaterial.texture = 0;

    defaultMaterial.texScaleU = 1;
    defaultMaterial.texScaleV = 1;

    // Exists
    if(mObject->vertices.size() > 1){
        // Iterate v
        for(int i=0; i<v.rows(); i++){
            OBJVertex& vertex = mObject->vertices[i];
            for(int j=0; j<v.cols(); j++){
                vertex.coords[j] = v(i,j);
            }
        }

        // Iterate f
        for(int i=0; i<f.rows(); i++){
            OBJFace& face = mObject->faces[i];;
            face.material = "";
            for(int j=0; j<f.cols(); j++){
                OBJFaceItem& item = face.items[j];
                item.vertexIndex  = 0;
                item.normalIndex  = 0;
                item.textureIndex = 0;
                item.vertexIndex = f(i,j); // faces are 0 based
            }
        }
    }else{
        // Iterate v
        for(int i=0; i<v.rows(); i++){
            OBJVertex vertex;
            for(int j=0; j<v.cols(); j++){
                vertex.coords[j] = v(i,j);
            }
            mObject->vertices.push_back(vertex);
            mObject->computedNormals.push_back( OBJNormal() );
        }

        // Iterate f
        for(int i=0; i<f.rows(); i++){
            OBJFace face;
            face.material = "";
            for(int j=0; j<f.cols(); j++){
                OBJFaceItem item;
                item.vertexIndex  = 0;
                item.normalIndex  = 0;
                item.textureIndex = 0;
                item.vertexIndex = f(i,j)+1; // faces are 0 based
                face.items.push_back(item);
            }
            mObject->faces.push_back(face);
        }
    }

    // Compute Normals
    for(OBJFace& face : mObject->faces){
        OBJVertex a = mObject->vertices.at(face.items.at(0).vertexIndex);
        OBJVertex b = mObject->vertices.at(face.items.at(1).vertexIndex);
        OBJVertex c = mObject->vertices.at(face.items.at(2).vertexIndex);
        float x = (b.coords[1]-a.coords[1])*(c.coords[2]-a.coords[2])-(b.coords[2]-a.coords[2])*(c.coords[1]-a.coords[1]);
        float y = (b.coords[2]-a.coords[2])*(c.coords[0]-a.coords[0])-(b.coords[0]-a.coords[0])*(c.coords[2]-a.coords[2]);
        float z = (b.coords[0]-a.coords[0])*(c.coords[1]-a.coords[1])-(b.coords[1]-a.coords[1])*(c.coords[0]-a.coords[0]);

        float length = std::sqrt( x*x + y*y + z*z );

        x /= length;
        y /= length;
        z /= length;

        for (int i=0; i<face.items.size(); ++i)
        {
            if ( mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[0]==0 &&
                 mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[1]==0 &&
                 mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[2]==0  )
            {
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[0] =  x;
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[1] =  y;
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[2] =  z;
            }
            else
            {
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[0] =  mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[0]+x;
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[1] =  mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[1]+y;
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[2] =  mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[2]+z;
            }
        }
    }

    return true;
}

bool WObject::loadOBJFile( const std::string& data_path, const std::string& mesh_filename, const std::string& material_filename )
{
    clearOBJFile();
    mDataPath = data_path;

    std::ifstream file_mesh;
    std::ifstream file_material;

    file_mesh.open( (data_path+mesh_filename).c_str() );
    if ( !material_filename.empty() )
        file_material.open( (data_path+material_filename).c_str() );

    if ( !file_mesh.good() || (!material_filename.empty() && !file_material.good()) ){
        error("Failed to load file", __LINE__, __FUNCTION__, __FILE__);
        return false;
    }

    std::string line;

    OBJMaterial& defaultMaterial = mObject->materials[""];
    defaultMaterial.Ka[0] = 1;
    defaultMaterial.Ka[1] = 1;
    defaultMaterial.Ka[2] = 1;
    defaultMaterial.Ka[3] = 1;

    defaultMaterial.Kd[0] = 1;
    defaultMaterial.Kd[1] = 1;
    defaultMaterial.Kd[2] = 1;
    defaultMaterial.Kd[3] = 1;

    defaultMaterial.Ks[0] = 1;
    defaultMaterial.Ks[1] = 1;
    defaultMaterial.Ks[2] = 1;
    defaultMaterial.Ks[3] = 1;

    defaultMaterial.texture = 0;

    defaultMaterial.texScaleU = 1;
    defaultMaterial.texScaleV = 1;

    if ( !material_filename.empty() )
    {
        while ( !file_material.eof() )
        {
            std::getline( file_material, line );
            //std::cout << line << std::endl;
            processMaterialLine( line );
        }
        file_material.close();
    }
    std::cout << "done mat" << std::endl;

    while ( !file_mesh.eof() )
    {
        std::getline( file_mesh, line );
        processMeshLine( line );
    }
    file_mesh.close();
    std::cout << "done mesh" << std::endl;

    return true;
}


void WObject::processMaterialLine( const std::string& line )
{
    if ( line.find("#")==0 ) // A comment
    {
        return;
    }
    else if ( line.find("newmtl ")==0 ) // A new material
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=2 )
            return;

        mCurrentMaterial = tokens[1];
    }
    else if ( line.find("Ka ")==0 ) // Ambient color
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=4 )
            return;

        OBJMaterial& material = mObject->materials[mCurrentMaterial];

        material.Ka[0] = std::strtod(tokens.at(1).c_str(), 0);
        material.Ka[1] = std::strtod(tokens.at(2).c_str(), 0);
        material.Ka[2] = std::strtod(tokens.at(3).c_str(), 0);
    }
    else if ( line.find("Kd ")==0 ) // Diffuse color
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=4 )
            return;

        OBJMaterial& material = mObject->materials[mCurrentMaterial];

        material.Kd[0] = std::strtod(tokens.at(1).c_str(), 0);
        material.Kd[1] = std::strtod(tokens.at(2).c_str(), 0);
        material.Kd[2] = std::strtod(tokens.at(3).c_str(), 0);
    }
    else if ( line.find("Ks ")==0 ) // Specular color
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=4 )
            return;

        OBJMaterial& material = mObject->materials[mCurrentMaterial];

        material.Ks[0] = std::strtod(tokens.at(1).c_str(), 0);
        material.Ks[1] = std::strtod(tokens.at(2).c_str(), 0);
        material.Ks[2] = std::strtod(tokens.at(3).c_str(), 0);
    }
    else if ( line.find("map_Kd ")==0 ) // Texture mapping
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        OBJMaterial& material = mObject->materials[mCurrentMaterial];

        if ( tokens.size()==2 )
        {
            loadTexture( mDataPath+tokens[1], false );
            material.texture   = getTexture( mDataPath+tokens[1] );
            material.texScaleU = 1;
            material.texScaleV = 1;
        }
        else if ( tokens.size()==6)
        {
            loadTexture( mDataPath+tokens[5], false );
            material.texture = getTexture( mDataPath+tokens[5] );
            material.texScaleU = std::strtod(tokens.at(2).c_str(), 0);
            material.texScaleV = std::strtod(tokens.at(3).c_str(), 0);
        }
        else
        {
            return;
        }
    }
    else // Whatever
    {
        return;
    }
}

void WObject::processMeshLine( const std::string& line )
{
    if ( line.find("#")==0 ) // A comment
    {
        return;
    }
    else if ( line.find("v ")==0 ) // A vertex
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=4 ){
            return;
        }

        OBJVertex vertex;
        vertex.coords[0] = strtod(tokens.at(1).c_str(), 0);
        vertex.coords[1] = strtod(tokens.at(2).c_str(), 0);
        vertex.coords[2] = strtod(tokens.at(3).c_str(), 0);

        mObject->vertices.push_back(vertex);
        mObject->computedNormals.push_back( OBJNormal() );
    }
    else if ( line.find("usemtl ")==0 ) // A material usage
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=2 ) return;

        mCurrentMaterial = tokens[1];
        std::cout << mCurrentMaterial << std::endl;
    }
    else if ( line.find("f ")==0 ) // A face
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        OBJFace face;

        for ( int i=1; i<tokens.size(); ++i ) // Each item of the face
        {
            OBJFaceItem item;

            item.vertexIndex  = 0;
            item.normalIndex  = 0;
            item.textureIndex = 0;

            std::vector<std::string> items;
            split( tokens.at(i), '/', items );

            switch (items.size())
            {
            case 1:
            {
                if (!items.at(0).empty()) item.vertexIndex  = std::strtol(items.at(0).c_str(), 0, 10);
                break;
            }
            case 2:
            {
                if (!items.at(0).empty()) item.vertexIndex  = std::strtol(items.at(0).c_str(), 0, 10);
                if (!items.at(1).empty()) item.textureIndex = std::strtol(items.at(1).c_str(), 0, 10);
                break;
            }
            case 3:
            {
                if (!items.at(0).empty()) item.vertexIndex  = std::strtol(items.at(0).c_str(), 0, 10);
                if (!items.at(1).empty()) item.textureIndex = std::strtol(items.at(1).c_str(), 0, 10);
                if (!items.at(2).empty()) item.normalIndex  = std::strtol(items.at(2).c_str(), 0, 10);
                break;
            }
            }

            face.items.push_back( item );
        }

        face.material = mCurrentMaterial;

        mObject->faces.push_back(face);

#ifdef USE_COMPUTED_NORMALS
        OBJVertex a = mObject->vertices.at(face.items.at(0).vertexIndex);
        OBJVertex b = mObject->vertices.at(face.items.at(1).vertexIndex);
        OBJVertex c = mObject->vertices.at(face.items.at(2).vertexIndex);
        float x = (b.coords[1]-a.coords[1])*(c.coords[2]-a.coords[2])-(b.coords[2]-a.coords[2])*(c.coords[1]-a.coords[1]);
        float y = (b.coords[2]-a.coords[2])*(c.coords[0]-a.coords[0])-(b.coords[0]-a.coords[0])*(c.coords[2]-a.coords[2]);
        float z = (b.coords[0]-a.coords[0])*(c.coords[1]-a.coords[1])-(b.coords[1]-a.coords[1])*(c.coords[0]-a.coords[0]);

        float length = std::sqrt( x*x + y*y + z*z );

        x /= length;
        y /= length;
        z /= length;

        for (int i=0; i<face.items.size(); ++i)
        {
            if ( mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[0]==0 &&
                 mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[1]==0 &&
                 mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[2]==0  )
            {
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[0] =  x;
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[1] =  y;
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[2] =  z;
            }
            else
            {
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[0] =  mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[0]+x;
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[1] =  mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[1]+y;
                mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[2] =  mObject->computedNormals.at(face.items.at(i).vertexIndex).coords[2]+z;
            }
        }
#endif
    }
    else if ( line.find("vn ")==0 ) // A normal
    {
#ifndef USE_COMPUTED_NORMALS
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=4 ) return;

        OBJNormal normal;
        normal.coords[0] = std::strtod(tokens.at(1).c_str(), 0);
        normal.coords[1] = std::strtod(tokens.at(2).c_str(), 0);
        normal.coords[2] = std::strtod(tokens.at(3).c_str(), 0);

        mObject->normals.push_back(normal);
#endif
    }
    else if ( line.find("vt ")==0 ) // A texture coordinate
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if(tokens.size() == 3){
            OBJTexture texture;
            texture.coords[0] = std::strtod(tokens.at(1).c_str(), 0 );
            texture.coords[1] = std::strtod(tokens.at(2).c_str(), 0 );
            texture.coords[2] = 1.0;

            mObject->textures.push_back(texture);
        }else if(tokens.size() == 4){
            OBJTexture texture;
            texture.coords[0] = std::strtod(tokens.at(1).c_str(), 0 );
            texture.coords[1] = std::strtod(tokens.at(2).c_str(), 0 );
            texture.coords[2] = std::strtod(tokens.at(3).c_str(), 0 );

            mObject->textures.push_back(texture);
        }else{
            std::cerr << "vt size error" << std::endl;
            throw;
        }

    }
    else // Whatever
    {
        return;
    }
}

WObject::WObject()
{
    listId = 0;
    mObject = std::make_shared<OBJObject>();
}

WObject::~WObject()
{

}

void WObject::print()
{
    OBJObject& object = *this->mObject.get();
    std::cout << "----------------" << std::endl;
    std::cout << "Vertices: " << object.vertices.size() << std::endl;
    std::cout << "Normals: " << object.normals.size() << std::endl;
    std::cout << "ComputedNormals: " << object.computedNormals.size() << std::endl;
    std::cout << "Textures: " << object.textures.size() << std::endl;
    std::cout << "Faces: " << object.faces.size() << std::endl;
    std::cout << "----------------" << std::endl;
}

GLuint WObject::getTexture(const std::string &filename)
{
    std::map<std::string,GLuint>::const_iterator it = textures.find( filename );

    if ( it==textures.end() )
    {
        std::cerr << "Warning: cannot retrieve non loaded texture \"" << filename << "\"" << std::endl;
        return 0;   // Texture not found
    }
    else
        return it->second; // Texture found
}

bool WObject::releaseTexture( const std::string& filename )
{
    // Retrieve texture id
    GLuint texture_id = getTexture(filename);

    if ( texture_id!=0 )
    {
        glDeleteTextures( 1, &texture_id ); // Texture found
        textures.erase( filename );
        return true;
    }
    else
    {
        std::cerr << "Warning: cannot release non loaded texture \"" << filename << "\"" << std::endl;
        return false;
    }
}

bool WObject::loadTexture( const std::string& filename, bool clamp )
{
    cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    if(image.empty()){
        std::cout << "image empty" << std::endl;
        return false;
    }else{
        cv::flip(image, image, 0);
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if ( clamp )
        {
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        }
        else // repeat
        {
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
        }
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        //glGenerateMipmap(GL_TEXTURE_2D);

        glTexImage2D(GL_TEXTURE_2D,     // Type of texture
                     0,                 // Pyramid level (for mip-mapping) - 0 is the top level
                     GL_RGB,            // Internal colour format to convert to
                     image.cols,          // Image width  i.e. 640 for Kinect in standard mode
                     image.rows,          // Image height i.e. 480 for Kinect in standard mode
                     0,                 // Border width in pixels (can either be 1 or 0)
                     GL_BGR, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                     GL_UNSIGNED_BYTE,  // Image data type
                     image.ptr());        // The actual image data itself

        textures.insert( std::make_pair(filename, textureID) );
        std::cout << "loaded texture " << filename << std::endl;
        return true;
    }
}

WRender3D::WRender3D()
{
    try
    {

    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

WRender3D::~WRender3D()
{
    try
    {
        //glutLeaveMainLoop();
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

std::mutex eventMutex;
float xTran = 0.0;
float yTran = 0.0;
float zTran = -3.0;
float xRot = 0.0;
float yRot = 0.0;
float zRot = 0.0;
bool lPress = 0;
bool rPress = 0;
bool mPress = 0;
int lastX = 0;
int lastY = 0;

void mouseButton(const int button, const int state, const int x, const int y)
{
    lastX = x;
    lastY = y;
    if (button == 3 || button == 4) //mouse wheel
    {
        if (button == 3)  //zoom in
        {
            zTran += 0.1;
            log("Zoom in");
        }
        else  //zoom out
        {
            zTran -= 0.1;
            log("Zoom out");
        }
    }
    else
    {
        if (button == GLUT_LEFT_BUTTON)
        {
            lPress = !state;
            log("Click " + std::to_string(lPress));
        }
        else if (button  == GLUT_RIGHT_BUTTON)
        {
            rPress = !state;
            log("Click " + std::to_string(rPress));
        }
        else if (button == GLUT_MIDDLE_BUTTON)
        {
            mPress = !state;
            log("Click " + std::to_string(mPress));
        }
    }
    glutPostRedisplay();
}

void mouseMotion(const int x, const int y)
{
    int dx = x - lastX;
    int dy = y - lastY;

    // If button1 pressed, zoom in/out if mouse is moved up/down.
    if (lPress)
    {
        log("Drag: " + std::to_string(dx) + " " + std::to_string(dy));
        xRot = xRot + 8*dy;
        yRot = yRot + 8*dx;
    } else if (rPress)
    {
        xRot = xRot + 8*dy;
        zRot = zRot + 8*dx;
    } else if (mPress) {
        xTran = xTran + 0.01*dx;
        yTran = yTran - 0.01*dy;
    }
    lastX = x;
    lastY = y;
    glutPostRedisplay();
}

// this is the actual idle function
void idleFunc()
{
    glutPostRedisplay();
    glutSwapBuffers();
}

void keyPressed(const unsigned char key, const int x, const int y)
{
    try
    {
        //const std::lock_guard<std::mutex> lock{gKeypoints3D.mutex};
        log("KEY");
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void renderMain(void)
{
    glutSwapBuffers();
}

const std::vector<GLfloat> LIGHT_DIFFUSE{ 1.f, 1.f, 1.f, 1.f };  // Diffuse light
const std::vector<GLfloat> LIGHT_POSITION{ 1.f, 1.f, 1.f, 0.f };  // Infinite light location
const std::vector<GLfloat> COLOR_DIFFUSE{ 0.5f, 0.5f, 0.5f, 1.f };

void WRender3D::initializationOnThread()
{
    try
    {
        // OpenGL - Initialization
        std::cout << "Initializing.." << std::endl;
        char *myargv [1];
        int myargc=1;
        myargv[0]=strdup ("Window");
        glutInit(&myargc, myargv);
        glutInitDisplayMode(GLUT_DOUBLE); // Enable double buffered mode
        glutInitWindowSize(renderWidth, renderHeight);   // Set the window's initial width & height
        glutInitWindowPosition(50, 50); // Position the window's initial top-left corner
        glutCreateWindow("Window");          // Create window with the given title
        //glutHideWindow();

        glLightfv(GL_LIGHT0, GL_AMBIENT, LIGHT_DIFFUSE.data());
        glLightfv(GL_LIGHT0, GL_DIFFUSE, LIGHT_DIFFUSE.data());
        glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION.data());
        glEnable(GL_LIGHT0);
        glEnable(GL_LIGHTING);

        glEnable( GL_DEPTH_TEST );
        glShadeModel( GL_SMOOTH );
        glEnable( GL_CULL_FACE );
        glClearColor( 1, 1, 1, 1 );
        //glClearColor( 44./255., 44./255., 44./255., 1 );

        glutMouseFunc(mouseButton);
        glutMotionFunc(mouseMotion);
        glutKeyboardFunc(keyPressed);
    }
    catch (const std::exception& e)
    {
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void WRender3D::workOnThread()
{
    try
    {
        renderMutex.lock();
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glEnable(GL_LIGHTING);
        glShadeModel( GL_SMOOTH );
        glEnable( GL_TEXTURE_2D );

        glViewport( 0, 0, (float)renderWidth/1, (float)renderHeight/1. );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        gluPerspective( 60, (float)renderWidth/(float)renderHeight, 0.1, 10000. );
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        eventMutex.lock();
        glPushMatrix();
        glTranslatef(xTran,yTran,zTran);
        glRotatef(xRot / 16.0, 1.0, 0.0, 0.0);
        glRotatef(yRot / 16.0, 0.0, 1.0, 0.0);
        glRotatef(zRot / 16.0, 0.0, 0.0, 1.0);
        eventMutex.unlock();
        for(auto obj : wObjects){
            obj->render();
        }
        glPopMatrix();

        glutSwapBuffers();
        glutPostRedisplay();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        glutMainLoopEvent();
        renderMutex.unlock();
    }
    catch (const std::exception& e)
    {
        //this->stop();
        error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void WRender3D::addObject(std::shared_ptr<WObject> wObject){
    wObject->rebuild();
    wObjects.push_back(wObject);
}
}
