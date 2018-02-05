#ifndef CAFFE_UTIL_PROPOSAL_
#define CAFFE_UTIL_PROPOSAL_
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <utility>
#include <string>
#include <vector>
#include <algorithm>
#include <iosfwd>
#include <memory>
//#include <caffe/util/text_detector.hpp>

namespace caffe {
struct Anchors{
    int base_size_;
    int scales_num_;
    int ratios_num_;
    std::vector<float> scales_vec_;
    std::vector<float> ratios_vec_;
    cv::Mat anchors_;
    Anchors(){
        base_size_ = 16;
        scales_num_ = 3;
        ratios_num_ = 4;
        scales_vec_.resize(scales_num_);
        ratios_vec_.resize(ratios_num_);
        scales_vec_[0] = 8; scales_vec_[1]=16; scales_vec_[2] = 32;
        ratios_vec_[0] = 0.5; ratios_vec_[1] = 1.0; ratios_vec_[2] = 2.0; ratios_vec_[3] = 3.0;
        anchors_ = cv::Mat(scales_num_ * ratios_num_, 4, CV_32F);
    }
    void generate_anchors();
};

class Proposal{
 public:
  Proposal();
  void generate_proposals(const caffe::Blob<float>* cls_blob, 
                          const caffe::Blob<float>* bb_blob,
                          float im_wid, float im_hei, float im_scale);
  cv::Mat get_proposals(){
      return proposals_.clone();
  }
  //void Detect(const cv::Mat& img, std::vector<Box>& final_dets);

 //private:
  /*
  void WrapInputLayer(std::vector<cv::Mat>* input_channels, const cv::Mat& img);
  void Preprocess(const cv::Mat& img, cv::Mat& out_img);
  void retrieve_bboxes(const Blob<float>* rois_blob,
                       const std::vector<float>& deltas_vec,
                       std::vector<float>& out_boxes);
  */

 private:
  int min_size_;
  int feat_stride_;
  int pre_nms_topn_;
  int post_nms_topn_;
  float nms_thresh_;
  Anchors all_anchors_;
  cv::Mat proposals_;
  std::vector<int> roi_indices_;
  //cv::Mat nms_mask_;
  
};

}//end of namespace caffe
#endif
