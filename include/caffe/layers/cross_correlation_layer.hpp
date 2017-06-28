#ifndef CAFFE_CROSS_CORRELATION_LAYER_HPP_
#define CAFFE_CROSS_CORRELATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"
//#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class CrossCorrelationLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit CrossCorrelationLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param) {};
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CrossCorrelation"; }
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }
  /*
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
  */

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 //     const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe
#endif  // CAFFE_CROSS_CORRELATION_LAYER_HPP_
