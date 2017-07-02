#ifndef CAFFE_SIAMESE_LOGISTIC_LOSS_LAYER_HPP_
#define CAFFE_SIAMESE_LOGISTIC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the logistic loss as described in paper
 * Fully-Convolutional Siamese Networks for Object Tracking
 * loss = log(1 + exp(-yv))
 * y is label(+1 for positive sample, -1 for negative sample)
 * v is predicted value
 */
template <typename Dtype>
class SiameseLogisticLossLayer : public LossLayer<Dtype> {
 public:
  explicit SiameseLogisticLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SiameseLogisticLoss"; }

 protected:
  /// @copydoc MultinomialLogisticLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the siamese logistic loss error gradient w.r.t. the
   *        predictions.
   *  dLdv = -y/(1+exp(yv))
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   virtual inline int ExactNumBottomBlobs() const { return 3; }
   Blob<Dtype> eyv_;
};

}  // namespace caffe

#endif  // CAFFE_SIAMESE_LOGISTIC_LOSS_LAYER_HPP_
