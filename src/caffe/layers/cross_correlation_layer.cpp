#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define DEBUG_INFO
#undef DEBUG_INFO
namespace caffe {

template <typename Dtype>
void CrossCorrelationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 // std::cout << "Enter cross relation reshape function" << std::endl;
    //delete bottom size constraint
    //bottom[0]: feature map from last layer
    //bottom[1]: kernel 
  const int first_spatial_axis = this->channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + this->num_spatial_axes_)
      << "bottom num_axes may not change.";
  this->num_ = bottom[0]->count(0, this->channel_axis_);
  CHECK_EQ(bottom[0]->shape(this->channel_axis_), this->channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
//  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
//    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
//        << "All inputs must have the same shape.";
//  }

  // Shape the tops.
  this->bottom_shape_ = &bottom[0]->shape();
  this->compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (this->reverse_dimensions()) {
    this->conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    this->conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  this->col_offset_ = this->kernel_dim_ * this->conv_out_spatial_dim_;
  this->output_offset_ = this->conv_out_channels_ * this->conv_out_spatial_dim_ / this->group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  this->conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = this->conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < this->num_spatial_axes_ + 1; ++i) {
    if (this->reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(this->channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(this->channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  this->col_buffer_shape_.clear();
  this->col_buffer_shape_.push_back(this->kernel_dim_ * this->group_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    if (this->reverse_dimensions()) {
      this->col_buffer_shape_.push_back(this->input_shape(i + 1));
    } else {
      this->col_buffer_shape_.push_back(this->output_shape_[i]);
    }
  }
  this->col_buffer_.Reshape(this->col_buffer_shape_);
  this->bottom_dim_ = bottom[0]->count(this->channel_axis_);
  this->top_dim_ = top[0]->count(this->channel_axis_);
  this->num_kernels_im2col_ = this->conv_in_channels_ * this->conv_out_spatial_dim_;
  this->num_kernels_col2im_ = this->reverse_dimensions() ? this->top_dim_ : this->bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  this->out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1, this->out_spatial_dim_);
    this->bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(this->bias_multiplier_.count(), Dtype(1),
        this->bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void CrossCorrelationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* weight = bottom[1]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_);
    if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
}

template <typename Dtype>
void CrossCorrelationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = bottom[1]->cpu_data();
  Dtype* weight_diff = bottom[1]->mutable_cpu_diff();
  for (int i = 0; i < 1; ++i) {
    Dtype* top_diff = top[i]->mutable_cpu_diff();
    /*
    const Dtype* tdata = top[i]->cpu_diff();
    std::cout << " CROSS WEIGHT DIFF: " << std::endl;
    for (int j = 0; j < top[i]->count(); ++j) {
        std::cout << tdata[j] << "    ";
    }
    std::cout << std::endl << std::endl << std::endl;
    */
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0] || true) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }//end of for
#ifdef DEBUG_INFO
  if (this->layer_param().name() == "cross" || true) {
      const Dtype* top_diff_0 = top[0]->cpu_diff();
      const Dtype* bottom_diff_0 = bottom[0]->cpu_data();
      const Dtype* bottom_diff_1 = bottom[1]->cpu_data();

      Dtype sum_top_0 = 0.0, sum_bottom_0 = 0.0, sum_bottom_1 = 0.0;
      std::cout << "top diff: " << std::endl;
      for (int i = 0; i < top[0]->count(); ++i) {
          //std::cout << top_diff_0[i] << " ";
          sum_top_0 += top_diff_0[i];
      }
      std::cout << std::endl;
      std::cout << "bottom 0  diff: " << std::endl;
      for (int i = 0; i < 50; ++i) {
          std::cout << bottom_diff_0[i * 50] << " ";
          sum_bottom_0 += bottom_diff_0[i];
      }
      std::cout << std::endl;
      std::cout << "bottom 1  diff: " << std::endl;
      for (int i = 0; i < 50; ++i) {
          std::cout << bottom_diff_1[i * 50] << " ";
          sum_bottom_1 += bottom_diff_1[i];
      }
      std::cout << std::endl;
      //std::cout << this->layer_param().name() << " cross top 0 bottom 0 1 diff: ";
      //std::cout << sum_top_0 << " " << sum_bottom_0 << " " << sum_bottom_1 << std::endl;
  }
#endif
}

#ifdef CPU_ONLY
STUB_GPU(CrossCorrelationLayer);
#endif

INSTANTIATE_CLASS(CrossCorrelationLayer);
REGISTER_LAYER_CLASS(CrossCorrelation);

}  // namespace caffe
