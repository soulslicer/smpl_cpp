#ifndef TRACKBAR_HPP
#define TRACKBAR_HPP

#include <opencv2/opencv.hpp>

class DoubleTrackManager{
    struct DoubleTrack{
    public:

        int int_value = 0;
        double precision = 100;
        double max_value;
        double curr_value = 0;
        bool change = false;

        void setup(const std::string& field_name, const std::string& window_name, double max_value, double start_value = 0, double default_value = 0, unsigned precision = 100){
            int_value = default_value * precision;
            cv::createTrackbar(field_name, window_name, &int_value, max_value * precision, DoubleTrack::callback, this);
            this->precision = precision;
            this->max_value = max_value;
            cv::setTrackbarPos(field_name, window_name, (int)((max_value * precision)/2) + (int)(start_value*precision));
        }

        double getCurrVal(){
            change = false;
            return curr_value;
        }

        bool changeOccured(){
            return change;
        }

        static void callback(int, void* object){
            DoubleTrack* pObject = static_cast<DoubleTrack*>(object);
            double val = (pObject->int_value / pObject->precision) - pObject->max_value/2;
            if(pObject->curr_value != val) pObject->change = true;
            pObject->curr_value = val;
        }
    };

    std::map<std::string, DoubleTrack> tracks;
    std::string winname;
public:

    DoubleTrackManager(){

    }

    void setupWindow(std::string winname){
        this->winname = winname;
        cv::namedWindow(winname);
    }

    void addTrack(std::string name, double maxVal, double startVal = 0){
        tracks[name] = DoubleTrack();
        tracks[name].setup(name, winname, maxVal, startVal);
    }

    double getTrackValue(std::string name){
        return tracks[name].getCurrVal();
    }

    bool changeOccured(){
        bool change = false;
        for (auto& kv : tracks) {
            if(kv.second.changeOccured()) change = true;
        }
        for (auto& kv : tracks) {
            kv.second.change = false;
        }
        return change;
    }

    void spin(int ms = 15){
        cv::waitKey(ms);
    }

};

#endif
