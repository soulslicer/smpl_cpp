#ifndef OP_HPP
#define OP_HPP

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <caffe/caffe.hpp>

#define FLAGS_logging_level 3
#define FLAGS_output_resolution "-1x-1"
#define FLAGS_net_resolution "-1x368"
#define FLAGS_model_pose "COCO"
#define FLAGS_alpha_pose 0.6
#define FLAGS_scale_gap 0.3
#define FLAGS_scale_number 1
#define FLAGS_render_threshold 0.05
#define FLAGS_num_gpu_start 0
#define FLAGS_disable_blending false
#define FLAGS_model_folder CMAKE_MODELS

class OpenPose{
public:
    std::unique_ptr<op::PoseExtractorCaffe> poseExtractorCaffe;
    std::unique_ptr<op::PoseCpuRenderer> poseRenderer;
    std::unique_ptr<op::FrameDisplayer> frameDisplayer;
    std::unique_ptr<op::ScaleAndSizeExtractor> scaleAndSizeExtractor;

    OpenPose(){
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SetDevice(0);

        op::log("OpenPose Library Tutorial - Example 1.", op::Priority::High);
        // ------------------------- INITIALIZATION -------------------------
        // Step 1 - Set logging level
            // - 0 will output all the logging messages
            // - 255 will output nothing
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        // Step 2 - Read Google flags (user defined configuration)
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // Check no contradictory flags enabled
        if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
            op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
        if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
            op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
                      __LINE__, __FUNCTION__, __FILE__);
        // Logging
        op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        // Step 3 - Initialize all required classes
        scaleAndSizeExtractor = std::unique_ptr<op::ScaleAndSizeExtractor>(new op::ScaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap));

        poseExtractorCaffe = std::unique_ptr<op::PoseExtractorCaffe>(new op::PoseExtractorCaffe{poseModel, FLAGS_model_folder, FLAGS_num_gpu_start});

        poseRenderer = std::unique_ptr<op::PoseCpuRenderer>(new op::PoseCpuRenderer{poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
                                                                                                    (float)FLAGS_alpha_pose});
        frameDisplayer = std::unique_ptr<op::FrameDisplayer>(new op::FrameDisplayer{"OpenPose Tutorial - Example 1", outputSize});


        // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
        poseExtractorCaffe->initializationOnThread();
        poseRenderer->initializationOnThread();
    }

    op::Array<float> forward(const cv::Mat& inputImage, bool display = false){
        op::OpOutputToCvMat opOutputToCvMat;
        op::CvMatToOpInput cvMatToOpInput;
        op::CvMatToOpOutput cvMatToOpOutput;
        if(inputImage.empty())
            op::error("Could not open or find the image: ", __LINE__, __FUNCTION__, __FILE__);
        const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
        // Step 2 - Get desired scale sizes
        std::vector<double> scaleInputToNetInputs;
        std::vector<op::Point<int>> netInputSizes;
        double scaleInputToOutput;
        op::Point<int> outputResolution;
        std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
            = scaleAndSizeExtractor->extract(imageSize);
        // Step 3 - Format input image to OpenPose input and output formats
        const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
        // Step 4 - Estimate poseKeypoints
        poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
        const auto poseKeypoints = poseExtractorCaffe->getPoseKeypoints();

        if(display){
            auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
            // Step 5 - Render poseKeypoints
            poseRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);
            // Step 6 - OpenPose output format to cv::Mat
            auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

            // ------------------------- SHOWING RESULT AND CLOSING -------------------------
            // Step 1 - Show results
            frameDisplayer->displayFrame(outputImage, 0); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
            // Step 2 - Logging information message
            op::log("Example 1 successfully finished.", op::Priority::High);
            // Return successful message
        }

        return poseKeypoints;
    }
};

#endif
