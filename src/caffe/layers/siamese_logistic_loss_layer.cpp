#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/siamese_logistic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define DEBUG_INFO
#undef DEBUG_INFO

namespace caffe {

template <typename Dtype>
void SiameseLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  int count = bottom[0]->count();
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  eyv_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SiameseLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //bottom[1]->Reshape(bottom[0]->count(), 1, 1, 1);
  Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* weight_data = bottom[2]->cpu_data();
#ifdef DEBUG_INFO
  std::cout << "bottom data: " << std::endl;
  for (int i = 0; i < bottom[0]->count(); ++i){
      std::cout << i << ": " << bottom_data[i]
                << "    "; 
  }
  std::cout << std::endl;
#endif
#if 0
  std::cout << "label data: " << std::endl;
  for (int i = 0; i < bottom[0]->count(); ++i){
      std::cout <<  i << ": " << bottom_label[i]
                << "    "; 
  }
  std::cout << std::endl;
#endif 

//  Dtype* yv_data = new Dtype[bottom[0]->count()];
  Dtype* yv_data = eyv_.mutable_cpu_data();
  caffe_mul<Dtype>(bottom[0]->count(), bottom_data, bottom_label, yv_data);
  caffe_scal<Dtype>(bottom[0]->count(), (Dtype)(-1.0), yv_data);
  caffe_exp<Dtype>(bottom[0]->count(), yv_data, yv_data);
  //caffe_add_scalar<Dtype>(bottom[0]->count(), (Dtype)1.0, yv_data);

  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
    loss += (weight_data[i] * log(yv_data[i] + 1));
  }
  top[0]->mutable_cpu_data()[0] = (Dtype)(1.0) * loss / bottom[0]->num();
  //delete [] yv_data;
}

template <typename Dtype>
void SiameseLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_data = bottom[0]->mutable_cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    const Dtype* weight_data = bottom[2]->cpu_data();

    //Dtype* yv_data = new Dtype[bottom[0]->count()];
    const Dtype* yv_data = eyv_.cpu_data();
    //caffe_mul<Dtype>(bottom[0]->count(), bottom_data, bottom_label, yv_data);
    //caffe_exp<Dtype>(bottom[0]->count(), yv_data, yv_data);
    //caffe_add_scalar<Dtype>(bottom[0]->count(), (Dtype)1.0, yv_data);
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = top[0]->cpu_diff()[0];
    for (int i = 0; i < bottom[0]->count(); ++i) {
      Dtype label = bottom_label[i];
      bottom_diff[i] = weight_data[i] * (Dtype)(-1.0) * label * yv_data[i] / (yv_data[i] + 1) / bottom[0]->num();
      //std::cout << "label: " << label << "  bottom data: " << bottom_data[i] <<  "  diff: " << bottom_diff[i] << std::endl;
    }
    //delete [] yv_data;
  }
#if 0
  std::cout << "bottom diff: " << std::endl;
  for (int i = 0; i < bottom[0]->count(); ++i){
      std::cout << i << ": " << bottom_diff[i]
                << "    "; 
  }
  std::cout << std::endl;
#endif

#if 0
  std::cout << "BACK DIFF: " << std::endl;
  const Dtype* tdiff = bottom[0]->cpu_diff();
  Dtype sum = 0.0;
  for (int i = 0; i < bottom[0]->count(); ++i){
      sum += (tdiff[i] * tdiff[i]);
      std::cout << tdiff[i] << "    " ;
  }
  std::cout << "loss diff: "  << std::sqrt(sum) << std::endl;
#endif
}

INSTANTIATE_CLASS(SiameseLogisticLossLayer);
REGISTER_LAYER_CLASS(SiameseLogisticLoss);

}  // namespace caffe
