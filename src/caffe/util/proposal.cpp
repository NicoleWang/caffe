#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <caffe/util/proposal.hpp>
#include "caffe/util/nms.hpp"
//#include <caffe/util/rfcn_detector.hpp>
//#include <caffe/util/text_detector.hpp>
//#define CPU_ONLY

#define ROUND(x) ((int)((x) + (float)0.5))

using std::max;
using std::min;
namespace caffe {
void Anchors::generate_anchors(){
    //base box's width & height & center location
    const float base_area = base_size_ * base_size_;
    const float center = 0.5*(base_size_ - 1.0);
    //std::cout << "base area: " << base_area << " " << "center: " << center << std::endl;

    //enumerate all transformed boxes
    int cnt = 0;
    for (int i = 0; i < ratios_num_; ++i) {
        // transformed width & height for given ratio factors
        float ratio_w = ROUND(sqrt(base_area / ratios_vec_[i]));
        float ratio_h = ROUND(ratio_w * ratios_vec_[i]);
        //std::cout << "rw: " << ratio_w << " " << "rh: " << ratio_h << std::endl;
        for (int j = 0; j < scales_num_; ++j) {
            //transformed width & height for given scale factors
            float scale_w = 0.5 * (ratio_w * scales_vec_[j] - 1.0);
            float scale_h = 0.5 * (ratio_h * scales_vec_[j] - 1.0);
            //std::cout << "sw: " << scale_w << " " << "sh: " << scale_h << std::endl;

            //(x1, y1, x2, y2) for transformed box
            anchors_.at<float>(cnt, 0) = center - scale_w;
            anchors_.at<float>(cnt, 1) = center - scale_h;
            anchors_.at<float>(cnt, 2) = center + scale_w;
            anchors_.at<float>(cnt, 3) = center + scale_h;
            //std::cout << anchors_.row(cnt);
            cnt++;
        }
    }
}

int transform_box(cv::Mat box, const float dx, const float dy, 
                  const float d_log_w, const float d_log_h, 
                  const float img_wid, const float img_hei,
                  const float min_box_W, const float min_box_H) {
    const float w = box.at<float>(0, 2) - box.at<float>(0, 0) + 1.0;
    const float h = box.at<float>(0, 3) - box.at<float>(0, 1) + 1.0;
    const float ctr_x = box.at<float>(0, 0) + 0.5 * w;
    const float ctr_y = box.at<float>(0, 1) + 0.5 * h;
    //std::cout << w << " " << h << " " << ctr_x << " " << ctr_y << std::endl;
    const float pred_ctr_x = dx * w + ctr_x;
    const float pred_ctr_y = dy * h + ctr_y;
    const float pred_w = exp(d_log_w) * w;
    const float pred_h = exp(d_log_h) * h;

    // update upper-left corner location
    box.at<float>(0, 0) = pred_ctr_x - 0.5 * pred_w;
    box.at<float>(0, 1) = pred_ctr_y - 0.5 * pred_h;
    // update lower-right corner location
    box.at<float>(0, 2) = pred_ctr_x + 0.5 * pred_w;
    box.at<float>(0, 3) = pred_ctr_y + 0.5 * pred_h;
    //std::cout << box << std::endl;

    // adjust new corner locations to be within the image region
    box.at<float>(0,0) = max(0.0f, min(box.at<float>(0,0), img_wid - 1.0f));
    box.at<float>(0,1) = max(0.0f, min(box.at<float>(0,1), img_hei - 1.0f));
    box.at<float>(0,2) = max(0.0f, min(box.at<float>(0,2), img_wid - 1.0f));
    box.at<float>(0,3) = max(0.0f, min(box.at<float>(0,3), img_hei - 1.0f));
    //recompute new width & height
    const float box_w = box.at<float>(0,2) - box.at<float>(0,0) + 1.0;
    const float box_h = box.at<float>(0,3) - box.at<float>(0,1) + 1.0;
    //check if new box's size >= threshold
    return (box_w >= min_box_H) * (box_h >=  min_box_H);
}


Proposal::Proposal():min_size_(16),feat_stride_(16), pre_nms_topn_(1000), post_nms_topn_(300),nms_thresh_(0.7){
    roi_indices_.resize(post_nms_topn_);
    all_anchors_.generate_anchors();
}

void Proposal::generate_proposals(const Blob<float>* cls_blob, const Blob<float>* bb_blob, float im_wid, float im_hei, float im_scale ){
    //get input data
    const float* cls_data = cls_blob->cpu_data();
    const float* bb_data = bb_blob->cpu_data();
    //TODO top data
    const int bottom_H = cls_blob->height();
    const int bottom_W = cls_blob->width();
    const int bottom_area = bottom_H * bottom_W;

    const float min_box_H = min_size_;
    const float min_box_W = min_size_;
    // number of all proposals = num_anchors * H * W
    cv::Mat anchors = all_anchors_.anchors_;
    //std::cout << "anchors in wangyuzhuo: " << std::endl << anchors << std::endl;
    const int num_proposals = anchors.rows * bottom_H * bottom_W;
    // number of top-n proposals before NMS
    const int pre_nms_topn = min(num_proposals, pre_nms_topn_);
    // number of final RoIs
    int num_rois = 0;

    //enumerate all proposals
    //(x1, y1, x2, y2, score) for each proposal
    // NOTE: for bottom, only foreground scores are passed
    const float*  score_data = cls_data + num_proposals;
    cv::Mat temp = cv::Mat::zeros(num_proposals, 5, CV_32F);
    int num_anchors = anchors.rows;
    int cnt = 0;
    std::cout << "wangyuzhuo" << std::endl;
    for (int h = 0; h < bottom_H; ++h) {
        for (int w = 0; w < bottom_W; ++w) {
            const float x = w * feat_stride_;
            const float y = h * feat_stride_;
            const float* p_box = bb_data + h * bottom_W + w;
            const float* p_score = score_data + h * bottom_W + w;
            for (int k = 0; k < num_anchors; ++k) {
                const float dx = p_box[(k * 4 + 0) * bottom_area];
                const float dy = p_box[(k * 4 + 1) * bottom_area];
                const float d_log_w = p_box[(k * 4 + 2) * bottom_area];
                const float d_log_h = p_box[(k * 4 + 3) * bottom_area];
                //std::cout << cnt << " " << x << " " << y << " " << anchors.row(k) << std::endl;
                temp.at<float>(cnt, 0) = x + anchors.at<float>(k, 0);
                temp.at<float>(cnt, 1) = y + anchors.at<float>(k, 1);
                temp.at<float>(cnt, 2) = x + anchors.at<float>(k, 2);
                temp.at<float>(cnt, 3) = y + anchors.at<float>(k, 3);
                //std::cout << temp.row(cnt) << std::endl;
                //std::cout << p_score[k * bottom_area] << std::endl;
                temp.at<float>(cnt, 4) = transform_box(temp.row(cnt), dx, dy, d_log_w, d_log_h, im_wid, im_hei, min_box_W, min_box_H) * p_score[k * bottom_area];
                cnt++;
            }
        }
    }
    //std::cout << "Proposals wangyuzhuo: "<< std::endl;
    //std::cout << temp << std::endl;
    cv::Mat all_scores = temp.col(4).clone();
    cv::Mat score_indices;
    cv::Mat sort_boxes = cv::Mat::zeros(num_proposals, 5, CV_32F);
    cv::sortIdx(all_scores, score_indices, CV_SORT_EVERY_COLUMN+CV_SORT_DESCENDING);
    for (int i = 0; i < all_scores.rows; ++i) {
        int idx = score_indices.at<int>(i, 0);
        temp.row(idx).copyTo(sort_boxes.row(i));
    }
    //std::cout << "wagyuzhuo after sort: " << sort_boxes << std::endl;
    nms_cpu<float>(pre_nms_topn, (float*)sort_boxes.data, roi_indices_.data(), &num_rois, 0, nms_thresh_, post_nms_topn_);
    /*
    for (int i = 0; i < roi_indices_.size(); ++i) {
        std::cout << roi_indices_[i] << " " ;
    }
    std::cout << std::endl;
    */
    proposals_ = cv::Mat(num_rois, 5, CV_32F);
    for (int i = 0; i < num_rois; ++i) {
        int idx = roi_indices_[i];
        proposals_.at<float>(i,0) = 0;
        proposals_.at<float>(i,1) = sort_boxes.at<float>(idx, 0);
        proposals_.at<float>(i,2) = sort_boxes.at<float>(idx, 1);;
        proposals_.at<float>(i,3) = sort_boxes.at<float>(idx, 2);;
        proposals_.at<float>(i,4) = sort_boxes.at<float>(idx, 3);;
    }
    //std::cout << proposals_ << std::endl;
}

}//end of namespace caffe
