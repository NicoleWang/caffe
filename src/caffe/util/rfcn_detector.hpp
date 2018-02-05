#ifndef CAFFE_UTIL_RFCN_DETECTOR_
#define CAFFE_UTIL_RFCN_DETECTOR_
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <utility>
#include <string>
#include <vector>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <caffe/util/text_detector.hpp>

namespace caffe {
/*
struct Box{
    float x;
    float y;
    float w;
    float h;
    float score;
    Box(){
        x = 0.0f;
        y = 0.0f;
        w = 0.0f;
        h = 0.0f;
        score = 0.0f;
    }
};

float iou(Box& b1, Box& b2);
void nms(std::vector<Box>& boxes, std::vector<Box>& out, float thresh);
*/

class RFCNDetector {
 public:
  RFCNDetector(const string& model_file,
           const string& weights_file,
           const int gpu_id);
  //std::vector<vector<float> > Detect(const cv::Mat& img);
  void Detect(const cv::Mat& img, std::vector<Box>& final_dets);

 private:
//  void SetMean();
  void WrapInputLayer(std::vector<cv::Mat>* input_channels, const cv::Mat& img);
  void Preprocess(const cv::Mat& img, cv::Mat& out_img);
  void retrieve_bboxes(const Blob<float>* rois_blob,
                       const std::vector<float>& deltas_vec,
                       std::vector<float>& out_boxes);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  unsigned int target_size_;
  unsigned int max_size_;
  cv::Scalar mean_;
  float image_scale_;
};

class PSRoI {
    public:
        PSRoI():spatial_scale_(0.0625),class_dim_(2), bbox_dim_(8), pooled_height_(7), pooled_width_(7),group_size_(7){};
        void do_psroi(const Blob<float>* class_map, const Blob<float>* bbox_map, const Blob<float>* rois, std::vector<float>& out_scores, std::vector<float>& out_bboxes);
    private:
        float spatial_scale_;
        //int output_dim_;
        int class_dim_;
        int bbox_dim_;
        int pooled_height_;
        int pooled_width_;
        int group_size_;
        
};

}//end of namespace caffe
#endif
