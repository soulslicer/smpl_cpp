#include "ObjectMesh.h"


ObjectMesh::ObjectMesh(){
    listId_ = 0;
    glutInitialised_ = false;
}

ObjectMesh::~ObjectMesh(){

}



bool ObjectMesh::startRenderer(){
    if(!glutInitialised_){
        cout << "GLUT not initialized" << endl;
        char *myargv [1];
        int myargc=1;
        myargv[0]=strdup ("Window");
        glutInit(&myargc, myargv);
        glutInitDisplayMode(GLUT_DOUBLE); // Enable double buffered mode
        glutInitWindowSize(renderWidth, renderHeight);   // Set the window's initial width & height
        glutInitWindowPosition(50, 50); // Position the window's initial top-left corner
        glutCreateWindow("Window");          // Create window with the given title
        glutHideWindow();
        glutInitialised_ = true;

        glEnable( GL_DEPTH_TEST );
        cout << "GL_DEPTH_TEST" << endl;
        glShadeModel( GL_SMOOTH );
        cout << "GL_SMOOTH" << endl;
        glEnable( GL_CULL_FACE );
        cout << "GL_CULL_FACE" << endl;
        //glClearColor( 1, 1, 1, 1 );
        glClearColor( 44./255., 44./255., 44./255., 1 );
        cout << "GL Clear Color" << endl;
    }
    cout << "GLUT initalized" << endl;

    return true;
    //glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

bool ObjectMesh::clearOBJFile(){
    if (listId_ != 0)
        glDeleteLists( listId_, 1 );
    object_.clear();

    std::vector<std::string> texKeys;
    for(std::map<std::string,GLuint>::iterator iterator = textures_.begin(); iterator != textures_.end(); iterator++) {
        texKeys.push_back(iterator->first);
    }
    for(int i=0; i<texKeys.size(); i++){
        release_texture(texKeys[i]);
    }
}

cv::Mat ObjectMesh::renderPerspective(DataCombiner::CameraObject& cameraObject, Eigen::Matrix4f cloudPose, bool debug){
    // I have no idea how the fuck this code works but it seems to work.

    // GL Stuff
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glDisable( GL_LIGHTING );
    glShadeModel( GL_FLAT );
    glDisable( GL_TEXTURE_2D );

    float width = cameraObject.rgbImage.size().width;
    float height = cameraObject.rgbImage.size().height;
    float fx = cameraObject.rgbCameraMatrix.at<float>(0,0);
    float cx = cameraObject.rgbCameraMatrix.at<float>(0,2);
    float fy = cameraObject.rgbCameraMatrix.at<float>(1,1);
    float cy = cameraObject.rgbCameraMatrix.at<float>(1,2);
    float xs = ((width/2)-cx);
    float ys = ((height/2)-cy);
    float fovy = (180.0 / M_PI) * (atan2(cy, fy)) * 2;

    Eigen::Matrix4f targetToRGB = PerceptionFunctions::eigenFromMat(cameraObject.transformMatrix["TargetToRGB"]);
    Eigen::Matrix4f finalTransform = targetToRGB * cloudPose.inverse();
    Eigen::VectorXf rpyxyz = PerceptionFunctions::getRPYXYZ(finalTransform);
    cout << "FOV: " << fovy << endl;
    cout << "Aspect: " << (float)width/(float)height << endl;
    cout << "XS YS: " << xs << " " << ys << endl;
    cout << rpyxyz.transpose() << endl;

    xs = -xs;
    ys = ys;
    glViewport(xs, ys, (float)renderWidth/1, (float)renderHeight/1. );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( fovy, (float)width/(float)height, 0.1, 100. );
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glPushMatrix();
    glTranslatef(rpyxyz(3),-rpyxyz(4),-rpyxyz(5));
    glRotatef(-rpyxyz(2), 0.0, 0.0, 1.0);
    glRotatef(-rpyxyz(1), 0.0, 1.0, 0.0);
    glRotatef(rpyxyz(0)+180, 1.0, 0.0, 0.0);
    glCallList( listId_ );
    glPopMatrix();

    // To CVMat
    cv::Mat out(cv::Size(renderWidth, renderHeight), CV_8UC3);
    glReadBuffer(GL_BACK);
    glReadPixels(0,0,renderWidth,renderHeight,GL_BGR,GL_UNSIGNED_BYTE,out.data);
    cv::Rect rect(renderWidth/2 - width/2, renderHeight/2 - height/2, width, height);
    cv::Mat cropped = out(rect);
    cv::Mat flipped;
    cv::flip(cropped, flipped, 0);
    return flipped;
}

cv::Mat ObjectMesh::renderOrtho(DataCombiner::CameraObject& cameraObject, float mult, float xl, float yl, Eigen::Matrix4f cloudPose, bool debug){
    // I have no idea how the fuck this code works but it seems to work.

    // GL Stuff
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glDisable( GL_LIGHTING );
    glShadeModel( GL_FLAT );
    glDisable( GL_TEXTURE_2D );

    float width = cameraObject.rgbImage.size().width;
    float height = cameraObject.rgbImage.size().height;

    Eigen::Matrix4f targetToRGB = PerceptionFunctions::eigenFromMat(cameraObject.transformMatrix["TargetToRGB"]);
    Eigen::Matrix4f finalTransform = targetToRGB * cloudPose.inverse();
    Eigen::VectorXf rpyxyz = PerceptionFunctions::getRPYXYZ(finalTransform);
    cout << "Aspect: " << (float)width/(float)height << endl;
    cout << rpyxyz.transpose() << endl;

    glViewport(0, 0, (float)xl*mult/1, (float)yl*mult/1. );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(-xl/2,xl/2,-yl/2,yl/2,0.1,100.);
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glPushMatrix();
    glTranslatef(rpyxyz(3),-rpyxyz(4),-rpyxyz(5));
    glRotatef(-rpyxyz(2), 0.0, 0.0, 1.0);
    glRotatef(-rpyxyz(1), 0.0, 1.0, 0.0);
    glRotatef(rpyxyz(0)+180, 1.0, 0.0, 0.0);
    glCallList( listId_ );
    glPopMatrix();

    // To CVMat
    cv::Mat out(cv::Size(renderWidth, renderHeight), CV_8UC3);
    glReadBuffer(GL_BACK);
    glReadPixels(0,0,renderWidth,renderHeight,GL_BGR,GL_UNSIGNED_BYTE,out.data);
    cv::Mat flipped;
    cv::flip(out, flipped, 0);
    cv::Rect rect(0, renderHeight - yl*mult, xl*mult, yl*mult);
    return flipped(rect);
}

cv::Mat ObjectMesh::getMat(){
    cv::Mat out(cv::Size(renderWidth, renderHeight), CV_8UC3);
    glReadBuffer(GL_BACK);
    glReadPixels(0,0,renderWidth,renderHeight,GL_BGR,GL_UNSIGNED_BYTE,out.data);
    cv::Mat flipped;
    cv::flip(out, flipped, 0);
    return flipped;
}

bool ObjectMesh::loadOBJFile( const std::string& data_path, const std::string& mesh_filename, const std::string& material_filename )
{
    clearOBJFile();
    dataPath_ = data_path;

    std::ifstream file_mesh;
    std::ifstream file_material;

    file_mesh.open( (data_path+mesh_filename).c_str() );
    if ( !material_filename.empty() )
        file_material.open( (data_path+material_filename).c_str() );

    if ( !file_mesh.good() || (!material_filename.empty() && !file_material.good()) )
        return false;

    std::string line;

    OBJMaterial& defaultMaterial = object_.materials[""];
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

    glEnable( GL_TEXTURE_2D );
    listId_ = glGenLists(1);
    glNewList( listId_, GL_COMPILE );

    glColor4f( 1.0f, 1.0f, 1.0f, 1.0f );

    std::cout << "building" << std::endl;
    build();
    std::cout << "built" << std::endl;

    glEndList();
    glDisable( GL_TEXTURE_2D );

    return true;
}

void ObjectMesh::processMaterialLine( const std::string& line )
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

        currentMaterial_ = tokens[1];
    }
    else if ( line.find("Ka ")==0 ) // Ambient color
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=4 )
            return;

        OBJMaterial& material = object_.materials[currentMaterial_];

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

        OBJMaterial& material = object_.materials[currentMaterial_];

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

        OBJMaterial& material = object_.materials[currentMaterial_];

        material.Ks[0] = std::strtod(tokens.at(1).c_str(), 0);
        material.Ks[1] = std::strtod(tokens.at(2).c_str(), 0);
        material.Ks[2] = std::strtod(tokens.at(3).c_str(), 0);
    }
    else if ( line.find("map_Kd ")==0 ) // Texture mapping
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        OBJMaterial& material = object_.materials[currentMaterial_];

        if ( tokens.size()==2 )
        {
            load_texture( dataPath_+tokens[1], false );
            material.texture   = get_texture( dataPath_+tokens[1] );
            material.texScaleU = 1;
            material.texScaleV = 1;
        }
        else if ( tokens.size()==6)
        {
            load_texture( dataPath_+tokens[5], false );
            material.texture = get_texture( dataPath_+tokens[5] );
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

void ObjectMesh::processMeshLine( const std::string& line )
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

        object_.vertices.push_back(vertex);
        object_.computedNormals.push_back( OBJNormal() );
    }
    else if ( line.find("usemtl ")==0 ) // A material usage
    {
        std::vector<std::string> tokens;
        split( line, ' ', tokens );

        if ( tokens.size()!=2 ) return;

        currentMaterial_ = tokens[1];
        std::cout << currentMaterial_ << std::endl;
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

        face.material = currentMaterial_;

        object_.faces.push_back(face);

#ifdef USE_COMPUTED_NORMALS
        OBJVertex a = object_.vertices.at(face.items.at(0).vertexIndex);
        OBJVertex b = object_.vertices.at(face.items.at(1).vertexIndex);
        OBJVertex c = object_.vertices.at(face.items.at(2).vertexIndex);
        float x = (b.coords[1]-a.coords[1])*(c.coords[2]-a.coords[2])-(b.coords[2]-a.coords[2])*(c.coords[1]-a.coords[1]);
        float y = (b.coords[2]-a.coords[2])*(c.coords[0]-a.coords[0])-(b.coords[0]-a.coords[0])*(c.coords[2]-a.coords[2]);
        float z = (b.coords[0]-a.coords[0])*(c.coords[1]-a.coords[1])-(b.coords[1]-a.coords[1])*(c.coords[0]-a.coords[0]);

        float length = std::sqrt( x*x + y*y + z*z );

        x /= length;
        y /= length;
        z /= length;

        for (int i=0; i<face.items.size(); ++i)
        {
            if ( object_.computedNormals.at(face.items.at(i).vertexIndex).coords[0]==0 &&
                 object_.computedNormals.at(face.items.at(i).vertexIndex).coords[1]==0 &&
                 object_.computedNormals.at(face.items.at(i).vertexIndex).coords[2]==0  )
            {
                object_.computedNormals.at(face.items.at(i).vertexIndex).coords[0] =  x;
                object_.computedNormals.at(face.items.at(i).vertexIndex).coords[1] =  y;
                object_.computedNormals.at(face.items.at(i).vertexIndex).coords[2] =  z;
            }
            else
            {
                object_.computedNormals.at(face.items.at(i).vertexIndex).coords[0] =  object_.computedNormals.at(face.items.at(i).vertexIndex).coords[0]+x;
                object_.computedNormals.at(face.items.at(i).vertexIndex).coords[1] =  object_.computedNormals.at(face.items.at(i).vertexIndex).coords[1]+y;
                object_.computedNormals.at(face.items.at(i).vertexIndex).coords[2] =  object_.computedNormals.at(face.items.at(i).vertexIndex).coords[2]+z;
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

        object_.normals.push_back(normal);
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

            object_.textures.push_back(texture);
        }else if(tokens.size() == 4){
            OBJTexture texture;
            texture.coords[0] = std::strtod(tokens.at(1).c_str(), 0 );
            texture.coords[1] = std::strtod(tokens.at(2).c_str(), 0 );
            texture.coords[2] = std::strtod(tokens.at(3).c_str(), 0 );

            object_.textures.push_back(texture);
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

void ObjectMesh::print(){
    std::cout << "----------------" << std::endl;
    std::cout << "Vertices: " << object_.vertices.size() << std::endl;
    std::cout << "Normals: " << object_.normals.size() << std::endl;
    std::cout << "ComputedNormals: " << object_.computedNormals.size() << std::endl;
    std::cout << "Textures: " << object_.textures.size() << std::endl;
    std::cout << "Faces: " << object_.faces.size() << std::endl;
    std::cout << "----------------" << std::endl;
}

void ObjectMesh::build()
{
    glEnable( GL_NORMALIZE );
    glEnable( GL_TEXTURE_2D );

    static float ambient [4];
    static float diffuse [4];
    static float specular[4];

    for ( int i=0; i<object_.faces.size(); ++i ) // Each face
    {
        OBJFace& face = object_.faces.at(i);
        OBJMaterial& material = object_.materials[face.material];

        //glMaterialfv( GL_FRONT, GL_AMBIENT,  material.Ka );
        //glMaterialfv( GL_FRONT, GL_DIFFUSE,  material.Kd );
        //glMaterialfv( GL_FRONT, GL_SPECULAR, material.Ks );

        glBindTexture( GL_TEXTURE_2D, material.texture );

        glBegin( GL_POLYGON );
        for ( int j=0; j<object_.faces.at(i).items.size(); ++j ) // Each vertex
        {
            OBJFaceItem& item = object_.faces.at(i).items.at(j);

#ifdef USE_COMPUTED_NORMALS
            glNormal3fv( object_.computedNormals.at(item.vertexIndex).coords );
#else
            glNormal3fv( object_.normals.at(item.vertexIndex).coords );
#endif

            if ( item.textureIndex > 0 ){
                glTexCoord2f( object_.textures.at(item.textureIndex).coords[0]*material.texScaleU,
                        object_.textures.at(item.textureIndex).coords[1]*material.texScaleV );
            }

            glVertex3fv( object_.vertices.at(item.vertexIndex).coords );

        }

        glEnd();
    }

    glDisable( GL_NORMALIZE );
}

bool ObjectMesh::load_texture( const std::string& filename, bool clamp )
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

        textures_.insert( std::make_pair(filename, textureID) );
        cout << "loaded texture " << filename << endl;
        return true;
    }
}

bool ObjectMesh::release_texture( const std::string& filename )
{
    // Retrieve texture id
    GLuint texture_id = get_texture(filename);

    if ( texture_id!=0 )
    {
        glDeleteTextures( 1, &texture_id ); // Texture found
        textures_.erase( filename );
        return true;
    }
    else
    {
        std::cerr << "Warning: cannot release non loaded texture \"" << filename << "\"" << std::endl;
        return false;
    }
}

GLuint ObjectMesh::get_texture( const std::string& filename )
{
    std::map<std::string,GLuint>::const_iterator it = textures_.find( filename );

    if ( it==textures_.end() )
    {
        std::cerr << "Warning: cannot retrieve non loaded texture \"" << filename << "\"" << std::endl;
        return 0;   // Texture not found
    }
    else
        return it->second; // Texture found
}

