#include <iostream>
using namespace std;

// python code/fit_3d.py $PWD --viz
// cd code;
// python smpl_webuser/hello_world/render_smpl.py

#include <renderer.hpp>
#include <eigen3/Eigen/Eigen>

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>

#include <cv.h>


inline float getpixel(const cv::Mat& in,
    std::size_t src_width, std::size_t src_height, unsigned y, unsigned x, int channel, int totalChannels)
{
    if (x < src_width && y < src_height)
        return in.at<float>((y * totalChannels * src_width) + (totalChannels * x) + channel);

    return 0;
}

inline float a1Eq(float d0, float d2, float d3){
    return -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
}

inline float a2Eq(float d0, float d2, float d3){
    return 1.0 / 2 * d0 + 1.0 / 2 * d2;
}

inline float a3Eq(float d0, float d2, float d3){
    return -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
}

void bicubicResize(const cv::Mat& in, cv::Mat& out, cv::Size newSize){
    if(in.channels() > 1) throw std::runtime_error("Only channel 1 supported");
    if(out.empty())
        out = cv::Mat(newSize, CV_32FC1, cv::Scalar(0));
    int totalChannels = 1;
    std::size_t src_width = in.size().width; std::size_t src_height = in.size().height;
    std::size_t dest_width = out.size().width; std::size_t dest_height = out.size().height;

    const float tx = float(src_width) / dest_width;
    const float ty = float(src_height) / dest_height;
    const int channels = totalChannels;
    const std::size_t row_stride = dest_width * channels;

    float C[5] = { 0 };

    for (int i = 0; i < dest_height; ++i)
    {
        for (int j = 0; j < dest_width; ++j)
        {
            const int x = int(tx * j);
            const int y = int(ty * i);
            const float dx = tx * j - (((float)x)+0.3);
            const float dy = ty * i - (((float)y)+0.3);


            for (int k = 0; k < totalChannels; ++k)
            {
                for (int jj = 0; jj < 4; ++jj)
                {
                    //out.at<uchar>(i * row_stride + j * channels + k) = getpixel(in, src_width, src_height, y, x, k, totalChannels);
                    const int z = y - 1 + jj;
                    float a0 = getpixel(in, src_width, src_height, z, x, k, totalChannels);
                    float d0 = getpixel(in, src_width, src_height, z, x - 1, k, totalChannels) - a0;
                    float d2 = getpixel(in, src_width, src_height, z, x + 1, k, totalChannels) - a0;
                    float d3 = getpixel(in, src_width, src_height, z, x + 2, k, totalChannels) - a0;
                    float a1 = a1Eq(d0, d2, d3);
                    float a2 = a2Eq(d0, d2, d3);
                    float a3 = a3Eq(d0, d2, d3);
                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = a1Eq(d0, d2, d3);
                    a2 = a2Eq(d0, d2, d3);
                    a3 = a3Eq(d0, d2, d3);
                    out.at<float>(i * row_stride + j * channels + k) = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
                    //out[i * row_stride + j * channels + k] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
                }
            }
        }
    }
}

//cv::Mat bicubicresize(const cv::Mat& in,
//    std::size_t src_width, std::size_t src_height, std::size_t dest_width, std::size_t dest_height, int totalChannels)
//{
//    cv::Mat out(cv::Size(dest_width, dest_height),CV_32FC1,cv::Scalar(0));
//    //std::vector<unsigned char> out(dest_width * dest_height * totalChannels);

//    const float tx = float(src_width) / dest_width;
//    const float ty = float(src_height) / dest_height;
//    const int channels = totalChannels;
//    const std::size_t row_stride = dest_width * channels;

//    float C[5] = { 0 };

//    for (int i = 0; i < dest_height; ++i)
//    {
//        for (int j = 0; j < dest_width; ++j)
//        {
//            const int x = int(tx * j);
//            const int y = int(ty * i);
//            const float dx = tx * j - x;
//            const float dy = ty * i - y;


//            for (int k = 0; k < totalChannels; ++k)
//            {
//                for (int jj = 0; jj < 4; ++jj)
//                {
//                    //out.at<uchar>(i * row_stride + j * channels + k) = getpixel(in, src_width, src_height, y, x, k, totalChannels);

//                    const int z = y - 1 + jj;
//                    float a0 = getpixel(in, src_width, src_height, z, x, k, totalChannels);
//                    float d0 = getpixel(in, src_width, src_height, z, x - 1, k, totalChannels) - a0;
//                    float d2 = getpixel(in, src_width, src_height, z, x + 1, k, totalChannels) - a0;
//                    float d3 = getpixel(in, src_width, src_height, z, x + 2, k, totalChannels) - a0;
//                    float a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
//                    float a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
//                    float a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
//                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

//                    d0 = C[0] - C[1];
//                    d2 = C[2] - C[1];
//                    d3 = C[3] - C[1];
//                    a0 = C[1];
//                    a1 = -1.0 / 3 * d0 + d2 -1.0 / 6 * d3;
//                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
//                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
//                    out.at<float>(i * row_stride + j * channels + k) = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
//                    //out[i * row_stride + j * channels + k] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
//                }
//            }
//        }
//    }

//    return out;
//}
int main(int argc, char *argv[])
{

    cv::Mat img = cv::imread("/home/ryaadhav/image_resize.png",0);
    img.convertTo(img, CV_32FC1);
    img*=0.005;
    cout << img.dims << endl;

    cv::Mat out;
    bicubicResize(img, out, cv::Size(img.size().width*2,img.size().height*2));

    cv::Mat out2;
    cv::resize(img, out2, cv::Size(img.size().width*2,img.size().height*2), 0,0,CV_INTER_CUBIC);


    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(out, &min, &max, &min_loc, &max_loc);
    cout << "Mine: " << max_loc << endl;

    cv::minMaxLoc(out2, &min, &max, &min_loc, &max_loc);
    cout << "OpenCV: " << max_loc << endl;

    cv::Mat x;
    cv::subtract(out, out2, x);

    cv::imshow("Mine",out);
    cv::imshow("OpenCV",out2);
    cv::imshow("diff",x);
    cv::waitKey(0);

    //std::vector<unsigned char> = bicubicresize()

    //cv::Mat out = bicubic(img, img.size().width*1,img.size().height*1);

    //    op::WRender3D render;
    //    render.initializationOnThread();
    //    std::shared_ptr<op::WObject> wObject1 = std::make_shared<op::WObject>();
    //    wObject1->loadOBJFile("/home/ryaadhav/project/","hello_smpl.obj","");
    //    render.addObject(wObject1);
    //    while(1){
    //        //wObject1->rebuild();
    //        render.workOnThread();
    //    }
}

