#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <caffe/util/rfcn_detector.hpp>
#include <caffe/util/text_detector.hpp>
#include <caffe/util/proposal.hpp>
#define CPU_ONLY
using std::max;
using std::min;
namespace caffe {
    void PSRoI::do_psroi(const Blob<float>* class_blob, const Blob<float>* bbox_blob, const Blob<float>* rois_blob, std::vector<float>& out_scores, std::vector<float>& out_bboxes) {
        //sanity checking
        /*
        std::cout << "class feat shape: " << class_blob->num() << " " << class_blob->channels() << " " << class_blob->height() << " " << class_blob->width() << std::endl;
        std::cout << "bbox feat shape: " << bbox_blob->num() << " " << bbox_blob->channels() << " " << bbox_blob->height() << " " << bbox_blob->width() << std::endl;
        std::cout << "roi feat shape: " << rois_blob->num() << " " << rois_blob->channels() << " " << rois_blob->height() << " " << rois_blob->width() << std::endl;
        */
        CHECK_EQ(class_blob->width(), bbox_blob->width())
            << "class blob and bbox blob must have same width";
        CHECK_EQ(class_blob->height(), bbox_blob->height())
            << "class blob and bbox blob must have same height"; 

        int height = class_blob->height();
        int width = class_blob->width();
        int roi_num = rois_blob->num();
        out_scores.resize(roi_num * class_dim_/2);
        out_bboxes.resize(roi_num * bbox_dim_/2);
        //std::vector<float> rfcn_scores(roi_num * class_dim_, 0);
        //std::vector<float> rfcn_bboxes(roi_num * bbox_dim_, 0);
        const float* class_map = class_blob->cpu_data();
        const float* bbox_map = bbox_blob->cpu_data();
        const float* rois = rois_blob->cpu_data();

        // get roi scores and bbox deltas for each roi
        for (unsigned int i = 0; i < roi_num; ++i) {
            std::vector<float> roi_score(2, 0.0);
            std::vector<float> roi_bbox(4, 0.0);
            const float* rois_data = rois + i*5;
            float roi_start_w = round(rois_data[1]) * spatial_scale_;
            float roi_start_h = round(rois_data[2]) * spatial_scale_;
            float roi_end_w = round(rois_data[3] + 1.0) * spatial_scale_;
            float roi_end_h = round(rois_data[4] + 1.0) * spatial_scale_;
            float roi_width = max(roi_end_w - roi_start_w, (float)0.1);
            float roi_height = max(roi_end_h - roi_start_h, (float)0.1);
            float bin_size_h = roi_height / pooled_height_;
            float bin_size_w = roi_width / pooled_width_;

            for (unsigned int j = 0; j < pooled_height_; ++j) {
                for (unsigned k = 0; k < pooled_width_; ++k) {
                    int hstart = floor(j*bin_size_h + roi_start_h);
                    int wstart = floor(k*bin_size_w + roi_start_w);
                    int hend = ceil((j+1)*bin_size_h + roi_start_h);
                    int wend = ceil((k+1)*bin_size_w + roi_start_w);

                    hstart = min(max(hstart, 0), height);
                    hend = min(max(hend, 0), height);
                    wstart = min(max(wstart, 0), width);
                    wend = min(max(wend, 0), width);
                    bool is_empty = (hend<= hstart) || (wend <= wstart);
                    if(is_empty) {
                        continue;
                    }
                    float bin_area = (hend - hstart)*(wend - wstart);
                    for (unsigned int cs = 0; cs < 2; ++cs) {
                        float bin_score = 0;
                        int score_c = (cs)*pooled_height_*pooled_width_ + j*pooled_width_ + k;
                        //std::cout << "cs=" <<cs<<" ";
                        const float* score_data = class_map + score_c*height*width; 
                        for (int h=hstart; h < hend; ++h){
                            for (int w=wstart; w < wend; ++w) {
                                int score_index = h*width + w;
                                //std::cout << score_data[score_index] << " ";
                                bin_score += score_data[score_index];
                            } 
                        }
                        //std::cout << std::endl;
                        roi_score[cs] += (bin_score / bin_area);
                    }
                    for (unsigned int cb =0; cb < 4; ++cb){
                        int bbox_c = (4+cb)*pooled_height_*pooled_width_ + j*pooled_width_ + k;
                        float bin_bbox = 0.0;
                        const float* bbox_data = bbox_map + bbox_c*width*height;
                        for (int h=hstart; h < hend; ++h){
                            for (int w=wstart; w < wend; ++w) {
                                int bbox_index = h*width + w;
                                bin_bbox += bbox_data[bbox_index];
                            }
                        }
                        roi_bbox[cb] += (bin_bbox/bin_area);
                    }

                }//end of k  
            }//end of j
            int patch_num = pooled_width_ * pooled_width_;
            //do softmax
            //std::cout << "roi score: " << roi_score[0] << " " << roi_score[1] << std::endl;
            roi_score[0] /= patch_num;
            roi_score[1] /= patch_num;
            float max_score = max(roi_score[0], roi_score[1]);
            roi_score[0] -= max_score;
            roi_score[1] -= max_score;
            float exp_sum = exp(roi_score[0] ) + exp(roi_score[0]);
            out_scores[i] = exp(roi_score[1])/exp_sum;
            for (int c = 0; c < 4; ++c){
                out_bboxes[i*4+c] = roi_bbox[c] / patch_num;
            }
        }//end of i
    }//end of function

RFCNDetector::RFCNDetector(const string& model_file,
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
    //CHECK_EQ(net_->num_outputs(), 3) << "Network should have exactly two outputs (scores, deltas and rois).";

    mean_ = cv::Scalar(102.9801, 115.9465, 122.7717);
    target_size_ = 256;
    max_size_ = 512;

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
void RFCNDetector::WrapInputLayer(std::vector<cv::Mat>* input_channels, const cv::Mat& img) {
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

void RFCNDetector::Preprocess(const cv::Mat& img, cv::Mat& out_img) {
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
    cv::Mat temp;
    resized_img.convertTo(temp, CV_32FC3);
    temp -= mean_;
    out_img = temp.clone();

    /* convert input image information to the input format of the network */
    Blob<float>* input_info = net_->input_blobs()[1];
    float* image_info = input_info->mutable_cpu_data();
    image_info[0] = resized_img.rows;
    image_info[1] = resized_img.cols;
    image_info[2] = ratio;
}



/*

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
*/
void trans_vec_to_boxes(std::vector<float>& scores,
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

void RFCNDetector::retrieve_bboxes(const Blob<float>* rois_blob,
                       const std::vector<float>& deltas_vec,
                       std::vector<float>& out_boxes) {
    int num_boxes = rois_blob->num();
    const float* rois = rois_blob->cpu_data();
    out_boxes.resize(4*num_boxes);
   
    for (int i = 0; i < num_boxes; ++i){
        const float* roi = rois + rois_blob->offset(i) + 1;
        float w = (roi[2] - roi[0]) / image_scale_ + 1.0f;
        float h = (roi[3] - roi[1]) / image_scale_ + 1.0f;
        float ctr_x = roi[0] / image_scale_ + 0.5f * w;
        float ctr_y = roi[1] / image_scale_ + 0.5f * h;

        float pred_ctr_x = deltas_vec[i*4] * w + ctr_x;
        float pred_ctr_y = deltas_vec[i*4+1] * h + ctr_y;
        float pred_w = std::exp(deltas_vec[i*4+2]) * w;
        float pred_h = std::exp(deltas_vec[i*4+3]) * h;

        //update upper-left corner location
        out_boxes[i * 4] = pred_ctr_x - 0.5f * pred_w;
        out_boxes[i * 4 + 1] = pred_ctr_y - 0.5f * pred_h;
        out_boxes[i * 4 + 2] = pred_ctr_x + 0.5f * pred_w;
        out_boxes[i * 4 + 3] = pred_ctr_y + 0.5f * pred_h;
    }
}

/*
void RFCNDetector::add_deltas_to_rois(const float* roi, 
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
*/

void  RFCNDetector::Detect(const cv::Mat& img, std::vector<Box>& final_dets) {
    cv::Mat post_img;
    //reshape and prepare image data
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
    std::cout << "ori shape: " << img.cols << " " << img.rows << std::endl
              << "new shape: " << post_img.cols << " " << post_img.rows << std::endl;
    std::cout << "image scale: " << image_scale_ << std::endl;
    input_image->Reshape(shape1);
    input_info->Reshape(shape2);
    
    /* forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels, post_img);

    //do net forward
    net_->Forward();

    //get all output blobs
    //const shared_ptr<Blob<float> > rois_blob = net_->blob_by_name("rois");
    const Blob<float>* rpn_cls_blob = net_->blob_by_name("rpn_cls_prob_reshape").get();
    const Blob<float>* rpn_bbox_blob = net_->blob_by_name("rpn_bbox_pred_new").get();
    const Blob<float>* img_info = net_->blob_by_name("im_info").get();
    const Blob<float>* bbox_blob  = net_->blob_by_name("rfcn_bbox").get();
    const Blob<float>* score_blob = net_->blob_by_name("text_rfcn_cls").get();
    //const Blob<float>* bbox_blob  = net_->output_blobs()[0];
    //const Blob<float>* score_blob = net_->output_blobs()[2];
    const float* bbox_data = bbox_blob->cpu_data();
    const float* score_data = score_blob->cpu_data();

    //get image infos
    const float* img_info_data = img_info->cpu_data();
    float img_hei = img_info_data[0];
    float img_wid = img_info_data[1];
    float img_scale = img_info_data[2];

    // get rpn proposals
    caffe::Proposal rpn_proposals;
    rpn_proposals.generate_proposals(rpn_cls_blob, rpn_bbox_blob, img_wid, img_hei, img_scale);
    cv::Mat props_mat = rpn_proposals.get_proposals();
    //std::cout << "Proposals: " << std::endl << props_mat << std::endl;
    Blob<float> rois_blob;
    std::vector<int> rois_shape(2);
    rois_shape[0] = props_mat.rows;
    rois_shape[1] = props_mat.cols;
    rois_blob.Reshape(rois_shape);
    float* rois_data = rois_blob.mutable_cpu_data();
    float* props_data =(float*) props_mat.data;
    for (int i = 0; i < rois_shape[0]*rois_shape[1]; ++i) {
        rois_data[i] = props_data[i];
    }


    std::vector<float> res_scores;
    std::vector<float> res_deltas;
    std::vector<float> res_bboxes;
    caffe::PSRoI ps_roi;
    /*
    for (unsigned int c = 0; c < 49; ++c) {
        std::cout << "score data:" << std::endl;
        const float* t_data = score_data + score_blob->offset(0,c);
        std::cout<<t_data[0]<<" "<<t_data[1]<<" "<<t_data[3]<<" "<<t_data[4]<<std::endl;
        t_data = score_data+score_blob->offset(0, c+49);
        std::cout<<t_data[0]<<" "<<t_data[1]<<" "<<t_data[3]<<" "<<t_data[4]<<std::endl;
    }
    */
    
    
    
    ps_roi.do_psroi(score_blob, bbox_blob, &rois_blob, res_scores, res_deltas);
    /*
    for (int i = 0; i < res_deltas.size(); ++i) {
        if (i%4 == 0) {
            std::cout << res_scores[i/4] << std::endl;
        }
        std::cout << res_deltas[i]  << " ";
        if (i%4 == 0) {
            std::cout << std::endl;
        }
    }
    */

    retrieve_bboxes(&rois_blob, res_deltas,  res_bboxes);
    std::vector<Box> new_boxes;
    trans_vec_to_boxes(res_scores, res_bboxes, new_boxes);

    //std::vector<Box> nms_boxes;
    nms(new_boxes, final_dets, 0.5);

    //cv::Mat vis_im = img.clone();
    //draw_boxes(vis_im, nms_boxes);

}
}//end of namespace caffe
