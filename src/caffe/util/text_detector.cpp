#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <caffe/util/text_detector.hpp>

namespace caffe {
float iou(Box& b1, Box& b2) {
    float xstart = std::max(b1.x, b2.x);
    float ystart = std::max(b1.y, b2.y);
    float xend = std::min(b1.x + b1.w - 1, b2.x + b2.w - 1);
    float yend = std::min(b1.y + b1.h - 1, b2.y + b2.h - 1);

    float delta_w = xend - xstart + 1;
    float delta_h = yend - ystart + 1;
    if (delta_w <= 0.0f || delta_h <= 0){
        return 0.0f;
    }

    float area1 = b1.w * b1.h;
    float area2 = b2.w * b2.h;
    float inter_area = delta_w * delta_h;

    return inter_area / (area1 + area2 - inter_area);

}

bool compare_box(const Box& b1, const Box& b2) {
    return (b1.score > b2.score ? true:false);
}

void nms(std::vector<Box>& boxes, std::vector<Box>& out, float thresh) {
    std::vector<bool> mask(boxes.size(), true);
    std::sort(boxes.begin(), boxes.end(), compare_box);

    int cnt = 0;
    for (unsigned int i = 0; i < boxes.size(); ++i) {
        if(!mask[i]){
            continue;
        }
        cnt++;
        for (unsigned int j = i + 1; j < boxes.size(); ++j){
            if(mask[j] && iou(boxes[i], boxes[j]) >= thresh) {
                mask[j] = false;
            }
        }
    }

    out.resize(cnt);
    cnt = 0;
    for (unsigned int i = 0; i < boxes.size(); ++i) {
        if (mask[i]){
            out[cnt] = boxes[i];
            cnt++;
        }
    }
}

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const int gpu_id) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else 
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_id);
#endif

    /* load trained caffe model */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);
    //printf("Net has %d inputs and %d outputs\n", net_->num_inputs(), net_->num_outputs());
    CHECK_EQ(net_->num_inputs(), 2) << "Network should have exactly two inputs (image data and image info)";
    CHECK_EQ(net_->num_outputs(), 2) << "Network should have exactly two outputs (scores and pred_boxes).";

    mean_ = cv::Scalar(102.9801, 115.9465, 122.7717);
    target_size_ = 600;
    max_size_ = 1000;

    Blob<float>* input_image = net_->input_blobs()[0];
    num_channels_ = input_image->channels();
    CHECK(num_channels_ == 3) << "Imput image must have 3 channels" ;
    //std::cout<< "input 1: " << input_image->shape_string() << std::endl;
    //std::cout<< "input 2: " << input_info->shape_string() << std::endl;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels, const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();

    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3); //Important, image data type must be float
    cv::split(img_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}

void Detector::Preprocess(const cv::Mat& img, cv::Mat& out_img) {
    /* convert input image to the input format of the network */
    cv::Mat sample;
    if (img.channels() == 4 && num_channels_ == 3) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1 && num_channels_ == 3) {
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    } else {
      sample = img; 
    }

    /* resize image's shortest side to 600 */
    int min_img_size = std::min(sample.rows, sample.cols);
    int max_img_size = std::max(sample.rows, sample.cols);
    float ratio = 1.0f * target_size_ / min_img_size;
    if (max_img_size * ratio > max_size_) {
        ratio = 1.0f * max_size_ / max_img_size;
    }
    int dst_wid = static_cast<int>(sample.cols * ratio);
    int dst_hei = static_cast<int>(sample.rows * ratio);
    image_scale_ = ratio;
    cv::Size dst_size(dst_wid, dst_hei);
    
    cv::Mat resized_img;
    cv::resize(sample, resized_img, dst_size);

    /* substract mean value */
    resized_img -= mean_;
    out_img = resized_img.clone();

    /* convert input image information to the input format of the network */
    Blob<float>* input_info = net_->input_blobs()[1];
    float* image_info = input_info->mutable_cpu_data();
    image_info[0] = resized_img.rows;
    image_info[1] = resized_img.cols;
    image_info[2] = ratio;
}

void get_predict_box(const float* roi, 
                     const float* delta, 
                     std::vector<float>& out,
                     const int idx,
                     float ratio=1.0f){
    float w = (roi[2] - roi[0]) / ratio + 1.0f;
    float h = (roi[3] - roi[1]) / ratio + 1.0f;
    float ctr_x = roi[0] / ratio + 0.5f * w;
    float ctr_y = roi[1] / ratio + 0.5f * h;

    //new center location according to gradient (dx, dy)
    float pred_ctr_x = delta[0] * w + ctr_x;
    float pred_ctr_y = delta[1] * h + ctr_y;

    //new width and height according to gradient d(log w), d(log h)
    float pred_w = std::exp(delta[2]) * w;
    float pred_h = std::exp(delta[3]) * h;

    //update upper-left corner location
    out[idx * 4] = pred_ctr_x - 0.5f * pred_w;
    out[idx * 4 + 1] = pred_ctr_y - 0.5f * pred_h;
    out[idx * 4 + 2] = pred_ctr_x + 0.5f * pred_w;
    out[idx * 4 + 3] = pred_ctr_y + 0.5f * pred_h;
}

void draw_boxes(cv::Mat& im, std::vector<float>& boxes, std::vector<float>& scores) {
    for (int i = 0; i < boxes.size() / 4; ++i) {
        cv::Point top_left((int)(boxes[i*4]), (int)(boxes[i*4 + 1]));
        cv::Point right_bottom((int)(boxes[i*4 + 2]), (int)(boxes[i*4 + 3]));
        if (scores[i] > 0.3f) {
            cv::rectangle(im, top_left, right_bottom, cv::Scalar(0,0,255));
        }
    }
    cv::imwrite("result.jpg", im);
}
void draw_boxes(cv::Mat& im, std::vector<Box>& boxes) {
    for (int i = 0; i < boxes.size(); ++i) {
        cv::Point top_left((int)(boxes[i].x), (int)(boxes[i].y));
        cv::Point right_bottom((int)(boxes[i].x + boxes[i].w - 1), (int)(boxes[i].y + boxes[i].h - 1));
        if (boxes[i].score > 0.3f) {
            cv::rectangle(im, top_left, right_bottom, cv::Scalar(0,0,255));
        }
    }
    cv::imwrite("result.jpg", im);
}
void transform_boxes(std::vector<float>& scores,
                     std::vector<float>& boxes,
                     std::vector<Box>& out) {
    out.resize(scores.size());
    for (unsigned int i = 0; i < scores.size(); ++i) {
        out[i].score = scores[i];
        float x1 = boxes[i*4];
        float y1 = boxes[i*4 + 1];
        float x2 = boxes[i*4 + 2];
        float y2 = boxes[i*4 + 3];
        out[i].x = x1;
        out[i].y = y1;
        out[i].w = x2 - x1 + 1.0f;
        out[i].h = y2 - y1 + 1.0f;
    }
}

void Detector::retrieve_bboxes(const shared_ptr<Blob<float> >& rois_blob,
                       const Blob<float>* deltas_blob,
                       const Blob<float>* scores_blob,
                       std::vector<float>& out_boxes,
                       std::vector<float>& out_scores) {
    int num_boxes = scores_blob->shape(0);
    const float* deltas = deltas_blob->cpu_data();
    const float* scores = scores_blob->cpu_data();
    const float* rois = rois_blob->cpu_data();
    out_boxes.resize(4*num_boxes);
    out_scores.resize(num_boxes);
   
    for (int i = 0; i < num_boxes; ++i){
        out_scores[i] = *(scores + scores_blob->offset(i) + 1);
        const float* cur_delta = deltas + deltas_blob->offset(i);
        const float* cur_roi = rois + rois_blob->offset(i) + 1;
        get_predict_box(cur_roi, cur_delta, out_boxes, i, image_scale_);
    }
}

void  Detector::Detect(const cv::Mat& img, std::vector<Box>& final_dets) {
    cv::Mat post_img;
    Preprocess(img, post_img);
    Blob<float>* input_image = net_->input_blobs()[0];
    Blob<float>* input_info = net_->input_blobs()[1];
    std::vector<int> shape1(4);
    shape1[0] = 1;
    shape1[1] = post_img.channels();
    shape1[2] = post_img.rows;
    shape1[3] = post_img.cols;
    std::vector<int> shape2(3);
    shape2[0] = 1;
    shape2[1] = 1;
    shape2[2] = 3;
    input_image->Reshape(shape1);
    input_info->Reshape(shape2);
    
    /* forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels, post_img);
    net_->ForwardPrefilled();

    const shared_ptr<Blob<float> > rois_blob = net_->blob_by_name("rois");

    Blob<float>* bbox_blob  = net_->output_blobs()[0];
    Blob<float>* score_blob = net_->output_blobs()[1];
    std::vector<float> res_scores;
    std::vector<float> res_bboxes;
    retrieve_bboxes(rois_blob, bbox_blob, score_blob, res_bboxes, res_scores);

    std::vector<Box> new_boxes;
    transform_boxes(res_scores, res_bboxes, new_boxes);

    //std::vector<Box> nms_boxes;
    nms(new_boxes, final_dets, 0.5);

    //cv::Mat vis_im = img.clone();
    //draw_boxes(vis_im, nms_boxes);

}
}//end of namespace caffe
