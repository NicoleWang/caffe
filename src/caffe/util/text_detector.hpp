#ifndef CAFFE_UTIL_TEXT_DETECTOR_
#define CAFFE_UTIL_TEXT_DETECTOR_
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <utility>
#include <string>
#include <vector>
#include <algorithm>
#include <iosfwd>
#include <memory>

namespace caffe {
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

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const int gpu_id);
  //std::vector<vector<float> > Detect(const cv::Mat& img);
  void Detect(const cv::Mat& img, std::vector<Box>& final_dets);

 private:
//  void SetMean();
  void WrapInputLayer(std::vector<cv::Mat>* input_channels, const cv::Mat& img);
  void Preprocess(const cv::Mat& img, cv::Mat& out_img);
  void retrieve_bboxes(const shared_ptr<Blob<float> >& rois_blob,
                       const Blob<float>* deltas_blob,
                       const Blob<float>* scores_blob,
                       std::vector<float>& out_boxes,
                       std::vector<float>& out_scores);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  unsigned int target_size_;
  unsigned int max_size_;
  cv::Scalar mean_;
  float image_scale_;
};

}//end of namespace caffe
#endif
