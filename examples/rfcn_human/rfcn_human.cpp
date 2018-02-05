#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <caffe/util/rfcn_detector.hpp>
#include <ctime>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

cv::Mat vis_boxes(const cv::Mat& im, std::vector<Box>& boxes){
    cv::Mat vis_im = im.clone();
    for (unsigned int i = 0; i < boxes.size(); ++i) {
        cv::Rect rect;
        cv::Scalar color(0, 0, 255);
        rect.x = static_cast<int>(boxes[i].x);
        rect.y = static_cast<int>(boxes[i].y);
        rect.width = static_cast<int>(boxes[i].w);
        rect.height = static_cast<int>(boxes[i].h);
        if (boxes[i].score > 0.3) {
            cv::rectangle(vis_im, rect, color);
        }
    }
    return vis_im;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " imglist_file resdir" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string resdir = argv[4];

  caffe::RFCNDetector detector(model_file, trained_file, 1);
  
  //process image one by one
  std::ifstream infile(argv[3]);
  std::string imagepath;
  clock_t start = clock();
  int img_num = 0;
  while (infile >> imagepath) {
      img_num++;
      cv::Mat img = cv::imread(imagepath);
      size_t s_pos = imagepath.rfind("/");
      std::string imname = imagepath.substr(s_pos, imagepath.length() - 1);
      //std::cout << img.cols << " " << img.rows << " " << img.channels() << std::endl; 
      CHECK(!img.empty()) << "Unable to decode image" << imagepath;
      std::vector<Box> dets;
      detector.Detect(img, dets);
      std::cout << "Processing " << imname << "    Detect " << dets.size() << " chars " << std::endl;
      //std::cout << "Detection time: " << duration * 1000 << " ms" << std::endl;
      cv::Mat vis_im = vis_boxes(img, dets);
      std::string savepath = resdir + "/" +imname;
      cv::imwrite(savepath, vis_im);
  }
  clock_t end = clock();
  double duration = (end - start) / (double) CLOCKS_PER_SEC;
  std::cout << "average detection time: " << duration * 1000 * 1.0 / img_num << " ms" <<std::endl;
}
