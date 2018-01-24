#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_reg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


using namespace cv;
using namespace std;
namespace caffe {

template <typename Dtype>
ImageRegDataLayer<Dtype>::~ImageRegDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageRegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  size_t pos;
  string filename;
  while (infile >> filename) {
    //std::cout << filename << std::endl;
    int l,t,r,b;
    infile >> l >> t >> r >> b;
    //std::cout << l << "  " << t << "  " << r << "  " << b << std::endl;
    Bbox bbox(l,t,r,b);
    lines_.push_back(std::make_pair(filename, bbox));
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  label_shape.push_back(4);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageRegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

void CropAndGetAnno(const cv::Mat& src, cv::Mat& dst, Bbox& tbox,
  Bbox& anno_box, vector<float>& anno_trans){
  cv::Mat src_broader;
  int iw = src.cols;
  int ih = src.rows;
  int left = max(0,-tbox.left);
  int right = max(0,tbox.right - iw);
  int top = max(0, -tbox.top);
  int bottom = max(0, tbox.bottom - ih);
  if(left + right + top + bottom > 0){
    copyMakeBorder(src,src_broader, top, bottom, left, right, 
      BORDER_CONSTANT, Scalar(128,128,128));
  } else {
    src_broader = src;
  }
  int crop_w = tbox.right - tbox.left;
  int crop_h = tbox.bottom - tbox.top;
  Rect roi = Rect(tbox.left + left, tbox.top + top, crop_w,
      crop_h);
  //LOG(INFO) << "ROI: "<<roi.x << " "<<roi.y<<" "<<roi.width <<" " <<roi.height;
  //LOG(INFO) << "Image: "<<src_broader.cols << " "<<src_broader.rows;

  src_broader(roi).copyTo(dst);
  anno_trans.push_back((anno_box.left-tbox.left)/float(crop_w));
  anno_trans.push_back((anno_box.top-tbox.top)/float(crop_h));
  anno_trans.push_back((tbox.right - anno_box.right)/float(crop_w));
  anno_trans.push_back((tbox.bottom - anno_box.bottom)/float(crop_h));
  //LOG(INFO) << "anno_trans: "<<anno_trans[0] << " "<<anno_trans[1]<<" "<<anno_trans[2] <<" " <<anno_trans[3];
  //Mat im_for_show;
  //dst.copyTo(im_for_show);
  //rectangle(im_for_show,Point(int(dst.cols*anno_trans[0]),int(dst.cols*anno_trans[1])),
  //  Point(int(dst.cols*(1-anno_trans[2])),int(dst.cols*(1-anno_trans[3]))),Scalar(0,255,0));
  //imshow("img",im_for_show);
  //waitKey(0);

}

float getRandFloat(){
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void CropAndAugment(const cv::Mat& im_in, cv::Mat& im_out, Bbox& anno_in,
  vector<float>& anno_out, int new_height, int new_width, const string& im_name = ""){
  int iw = im_in.cols;
  int ih = im_in.rows;
  /*
  int bh = anno_in.bottom - anno_in.top;
  int bw = anno_in.right - anno_in.left;
  int base_crop_size = max(bh,bw);
  float r = getRandFloat();
  int crop_size = int(base_crop_size*(1 + r));
  r = getRandFloat();
  int start_x = anno_in.left - int((crop_size - bw)*r);
  r = getRandFloat();
  int start_y = anno_in.top - int((crop_size - bh)*r);
  int end_x = start_x + crop_size;
  int end_y = start_y + crop_size;
  Mat tmp2;
  Bbox tbox(start_x, start_y, end_x, end_y);
  CHECK(crop_size > 1) << "crop_size error " << crop_size  << "ratio: " << r<<" "
    << im_name;
  std::cout << im_name << std::endl;
  CropAndGetAnno(im_in,tmp2,tbox,anno_in, anno_out);
  */
  anno_out.push_back(anno_in.left*1.0/iw);
  anno_out.push_back(anno_in.top*1.0/ih);
  anno_out.push_back(1.0 - anno_in.right*1.0/iw);
  anno_out.push_back(1.0 - anno_in.bottom*1.0/ih);
  //std::cout<< "iw: " << iw << " ih: "<< ih << " "<<  anno_out[0] << "  " << anno_out[1] << "   " << anno_out[2] << "  " << anno_out[3] << std::endl;
  resize(im_in, im_out, Size(new_width, new_height));
}


// This function is called on prefetch thread
template <typename Dtype>
void ImageRegDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  //CHECK(new_height==new_width);
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat tmp = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
    CHECK(tmp.data) << "Could not load " << lines_[lines_id_].first;

    cv::Mat cv_img;
    vector<float> anno;
    CropAndAugment(tmp, cv_img, lines_[lines_id_].second, anno, 
      new_height,new_width, lines_[lines_id_].first);
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    int begin_idx = 4*item_id;
    for( int i = 0; i < 4; i++){
      prefetch_label[begin_idx + i] = anno[i];
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageRegDataLayer);
REGISTER_LAYER_CLASS(ImageRegData);

}  // namespace caffe
#endif  // USE_OPENCV
