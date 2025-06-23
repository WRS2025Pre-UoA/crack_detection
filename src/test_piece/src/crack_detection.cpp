#include <random>

#include <filesystem>
#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/augment.h"
#include "constants.h"
#include "utils/common.h"
#include <string>


namespace fs = std::filesystem;


// Define the skeleton and color mappings
std::vector<std::vector<int>> skeleton = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7},
                                          {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

std::vector<cv::Scalar> posePalette = {
        cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51), cv::Scalar(255, 178, 102), cv::Scalar(230, 230, 0), cv::Scalar(255, 153, 255),
        cv::Scalar(153, 204, 255), cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255), cv::Scalar(102, 178, 255), cv::Scalar(51, 153, 255),
        cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102), cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
        cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255)
};

std::vector<int> limbColorIndices = {9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};
std::vector<int> kptColorIndices = {16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};



cv::Scalar generateRandomColor(int numChannels) {
    if (numChannels < 1 || numChannels > 3) {
        throw std::invalid_argument("Invalid number of channels. Must be between 1 and 3.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    cv::Scalar color;
    for (int i = 0; i < numChannels; i++) {
        color[i] = dis(gen); // for each channel separately generate value
    }

    return color;
}

std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels) {
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < class_names_num; i++) {
        cv::Scalar color = generateRandomColor(numChannels);
        colors.push_back(color);
    }
    return colors;
}

void plot_masks(cv::Mat img, std::vector<YoloResults>& result, std::vector<cv::Scalar> color,
    std::unordered_map<int, std::string>& names)
{
    cv::Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++)
    {
        float left, top;
        left = result[i].bbox.x;
        top = result[i].bbox.y;
        int color_num = i;
        int& class_idx = result[i].class_idx;
        rectangle(img, result[i].bbox, color[result[i].class_idx], 2);

        // try to get string value corresponding to given class_idx
        std::string class_name;
        auto it = names.find(class_idx);
        if (it != names.end()) {
            class_name = it->second;
        }
        else {
            std::cerr << "Warning: class_idx not found in names for class_idx = " << class_idx << std::endl;
            // then convert it to string anyway
            class_name = std::to_string(class_idx);
        }

        if (result[i].mask.rows && result[i].mask.cols > 0)
        {
            mask(result[i].bbox).setTo(color[result[i].class_idx], result[i].mask);
        }
        std::stringstream labelStream;
        labelStream << class_name << " " << std::fixed << std::setprecision(2) << result[i].conf;
        std::string label = labelStream.str();

    	cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
        cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        rectangle(img, rect_to_fill, color[result[i].class_idx], -1);

        putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
    }
    addWeighted(img, 0.6, mask, 0.4, 0, img); //add mask to src
    resize(img, img, img.size());
    //imshow("img", img);
    cv::waitKey();
}


//void plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const cv::Size& shape) {
void plot_keypoints(cv::Mat& image, const std::vector<YoloResults>& results, const cv::Size& shape) {

    int radius = 5;
    bool drawLines = true;

    if (results.empty()) {
        return;
    }

    std::vector<cv::Scalar> limbColorPalette;
    std::vector<cv::Scalar> kptColorPalette;

    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }

    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (const auto& res: results) {
        auto keypoint = res.keypoints;
        bool isPose = keypoint.size() == 51;  // numKeypoints == 17 && keypoints[0].size() == 3;
        drawLines &= isPose;

        // draw points
        for (int i = 0; i < 17; i++) {
            int idx = i * 3;
            int x_coord = static_cast<int>(keypoint[idx]);
            int y_coord = static_cast<int>(keypoint[idx + 1]);

            if (x_coord % shape.width != 0 && y_coord % shape.height != 0) {
                if (keypoint.size() == 3) {
                    float conf = keypoint[2];
                    if (conf < 0.5) {
                        continue;
                    }
                }
                cv::Scalar color_k = isPose ? kptColorPalette[i] : cv::Scalar(0, 0,
                                                                               255);  // Default to red if not in pose mode
                cv::circle(image, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
            }
        }
        // draw lines
        if (drawLines) {
            for (int i = 0; i < skeleton.size(); i++) {
                const std::vector<int> &sk = skeleton[i];
                int idx1 = sk[0] - 1;
                int idx2 = sk[1] - 1;

                int idx1_x_pos = idx1 * 3;
                int idx2_x_pos = idx2 * 3;

                int x1 = static_cast<int>(keypoint[idx1_x_pos]);
                int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
                int x2 = static_cast<int>(keypoint[idx2_x_pos]);
                int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

                float conf1 = keypoint[idx1_x_pos + 2];
                float conf2 = keypoint[idx2_x_pos + 2];

                // Check confidence thresholds
                if (conf1 < 0.5 || conf2 < 0.5) {
                    continue;
                }

                // Check if positions are within bounds
                if (x1 % shape.width == 0 || y1 % shape.height == 0 || x1 < 0 || y1 < 0 ||
                    x2 % shape.width == 0 || y2 % shape.height == 0 || x2 < 0 || y2 < 0) {
                    continue;
                }

                // Draw a line between keypoints
                cv::Scalar color_limb = limbColorPalette[i];
                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
            }
        }
    }
}

void segmented_object(const cv::Mat& img, const std::vector<YoloResults>& results){
   
}

void plot_results(cv::Mat img, std::vector<YoloResults>& results,
                  std::vector<cv::Scalar> color, std::unordered_map<int, std::string>& names,
                  const cv::Size& shape, std::string file_name
                  ) {


    int instance_counter = 0;

    cv::Mat mask = img.clone();

    int radius = 5;
    bool drawLines = true;

    auto raw_image_shape = img.size();
    std::vector<cv::Scalar> limbColorPalette;
    std::vector<cv::Scalar> kptColorPalette;

    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }

    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (const auto& res : results) {
        float left = res.bbox.x;
        float top = res.bbox.y;
        int color_num = res.class_idx;

        // Draw bounding box
        rectangle(img, res.bbox, color[res.class_idx], 2);

        // Try to get the class name corresponding to the given class_idx
        std::string class_name;
        auto it = names.find(res.class_idx);
        if (it != names.end()) {
            class_name = it->second;
        }
        else {
            std::cerr << "Warning: class_idx not found in names for class_idx = " << res.class_idx << std::endl;
            // Then convert it to a string anyway
            class_name = std::to_string(res.class_idx);
        }

        // Draw mask if available
        if (res.mask.rows && res.mask.cols > 0) {
            mask(res.bbox).setTo(color[res.class_idx], res.mask);

            // 元のマスク（0 or 1 の2値）を取得
            if (res.mask.rows > 0 && res.mask.cols > 0) {
                cv::Mat cropped_img = img(res.bbox).clone();
                cv::Mat mask = res.mask.clone();

                // マスクを255スケールに変換
                cv::Mat mask_bin;
                mask.convertTo(mask_bin, CV_8UC1, 255);

                // 輪郭抽出
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(mask_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                if (contours.empty()) return;

                // 最も大きな輪郭を対象にする
                size_t max_idx = 0;
                double max_area = 0.0;
                for (size_t i = 0; i < contours.size(); ++i) {
                    double area = cv::contourArea(contours[i]);
                    if (area > max_area) {
                        max_area = area;
                        max_idx = i;
                    }
                }
                std::vector<cv::Point> contour = contours[max_idx];

                // 四隅に近い点を抽出（最も端の点）
                cv::Point tl, tr, br, bl;
                double min_sum = 1e9, max_sum = -1e9;
                double min_diff = 1e9, max_diff = -1e9;

                for (const auto& pt : contour) {
                    int sum = pt.x + pt.y;
                    int diff = pt.x - pt.y;

                    if (sum < min_sum) { min_sum = sum; tl = pt; }      // top-left
                    if (sum > max_sum) { max_sum = sum; br = pt; }      // bottom-right
                    if (diff < min_diff) { min_diff = diff; bl = pt; }  // bottom-left
                    if (diff > max_diff) { max_diff = diff; tr = pt; }  // top-right
                }

                float shrink_ratio = 0.05f; // 3% 縮小

                // 四隅の点（前コードの tl, tr, br, bl）
                std::vector<cv::Point2f> quad_pts = { tl, tr, br, bl };

                // 中心点（重心）を求める
                cv::Point2f center(0.f, 0.f);
                for (const auto& pt : quad_pts) center += pt;
                center *= 1.0f / quad_pts.size();

                // 各点を中心に向かって移動
                std::vector<cv::Point> shrunk_quad;
                for (const auto& pt : quad_pts) {
                    cv::Point2f direction = center - pt;
                    cv::Point2f new_pt = pt + direction * shrink_ratio;
                    shrunk_quad.emplace_back(cv::Point(cvRound(new_pt.x), cvRound(new_pt.y)));
                }

                // 新しいマスク生成
                cv::Mat shrunk_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
                cv::fillConvexPoly(shrunk_mask, shrunk_quad, 255);

                //黒背景？
                // 切り抜き
                //cv::Mat segmented_shrunk;
                //cropped_img.copyTo(segmented_shrunk, shrunk_mask);

                //透明背景？
                //4チャンネル？
                cv::Mat segmented_shrunk(shrunk_mask.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0)); // 完全に透明な画像
                for (int y = 0; y < cropped_img.rows; ++y) {
                    for (int x = 0; x < cropped_img.cols; ++x) {
                        if (shrunk_mask.at<uchar>(y, x) > 0) {
                            cv::Vec3b bgr = cropped_img.at<cv::Vec3b>(y, x);
                            segmented_shrunk.at<cv::Vec4b>(y, x) = cv::Vec4b(bgr[0], bgr[1], bgr[2], 255); // 不透明
                        }
                    }
                }



                // 表示・保存
                fs::path output_folder = "results_cut";
                std::string extension = ".png";
                std::ostringstream oss;
                oss << file_name << "_" << instance_counter << extension;

                fs::path save_path = output_folder / oss.str();
                //cv::imshow("shrunk segmented", segmented_shrunk);
                cv::imwrite(save_path.string(), segmented_shrunk);

                instance_counter++;
            }
            
        }

        // Create label
        std::stringstream labelStream;
        labelStream << class_name << " " << std::fixed << std::setprecision(2) << res.conf;
        std::string label = labelStream.str();

        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
        cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        rectangle(img, rect_to_fill, color[res.class_idx], -1);
        putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);

        // Check if keypoints are available
        if (!res.keypoints.empty()) {
            auto keypoint = res.keypoints;
            bool isPose = keypoint.size() == 51;  // numKeypoints == 17 && keypoints[0].size() == 3;
            drawLines &= isPose;

            // draw points
            for (int i = 0; i < 17; i++) {
                int idx = i * 3;
                int x_coord = static_cast<int>(keypoint[idx]);
                int y_coord = static_cast<int>(keypoint[idx + 1]);

                if (x_coord % raw_image_shape.width != 0 && y_coord % raw_image_shape.height != 0) {
                    if (keypoint.size() == 3) {
                        float conf = keypoint[2];
                        if (conf < 0.5) {
                            continue;
                        }
                    }
                    cv::Scalar color_k = isPose ? kptColorPalette[i] : cv::Scalar(0, 0,
                                                                                  255);  // Default to red if not in pose mode
                    cv::circle(img, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
                }
            }
            // draw lines
            if (drawLines) {
                for (int i = 0; i < skeleton.size(); i++) {
                    const std::vector<int> &sk = skeleton[i];
                    int idx1 = sk[0] - 1;
                    int idx2 = sk[1] - 1;

                    int idx1_x_pos = idx1 * 3;
                    int idx2_x_pos = idx2 * 3;

                    int x1 = static_cast<int>(keypoint[idx1_x_pos]);
                    int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
                    int x2 = static_cast<int>(keypoint[idx2_x_pos]);
                    int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

                    float conf1 = keypoint[idx1_x_pos + 2];
                    float conf2 = keypoint[idx2_x_pos + 2];

                    // Check confidence thresholds
                    if (conf1 < 0.5 || conf2 < 0.5) {
                        continue;
                    }

                    // Check if positions are within bounds
                    if (x1 % raw_image_shape.width == 0 || y1 % raw_image_shape.height == 0 || x1 < 0 || y1 < 0 ||
                        x2 % raw_image_shape.width == 0 || y2 % raw_image_shape.height == 0 || x2 < 0 || y2 < 0) {
                        continue;
                    }

                    // Draw a line between keypoints
                    cv::Scalar color_limb = limbColorPalette[i];
                    cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
                }
            }
        }
    }

    // Combine the image and mask
    addWeighted(img, 0.6, mask, 0.4, 0, img);
}



int main()
{    
    std::string image_folder = "original_imgs";
    const std::string& modelPath = "checkpoints/best_n_ex_2.onnx";
    const std::string& onnx_provider = OnnxProviders::CPU;
    const std::string& onnx_logid = "yolov8_inference2";
    float mask_threshold = 0.5f;
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f;
    int conversion_code = cv::COLOR_BGR2RGB;

    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());
    std::vector<cv::Scalar> colors = generateRandomColors(model.getNc(), model.getCh());
    std::unordered_map<int, std::string> names = model.getNames();

    for (const auto& entry : fs::directory_iterator(image_folder)) {
        if (entry.is_regular_file()) {
            std::string img_path = entry.path().string();

            if (entry.path().extension() != ".JPG") {
            continue;
            }

            cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
            if (img.empty()) {
                std::cerr << "Error loading image: " << img_path << std::endl;
                continue;
            }

            std::cout << "Processing: " << img_path << std::endl;
            std::vector<YoloResults> objs = model.predict_once(img, conf_threshold, iou_threshold, mask_threshold, conversion_code);

            // 新しいファイル名作成
            fs::path output_path = entry.path().stem();
            //output_path += "-res";

            fs::path output_path_ex = output_path;

            fs::create_directories("results");

            output_path += entry.path().extension();
            //output_path = entry.path().parent_path() / output_path;
            output_path = fs::path("results") / output_path;

            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            cv::Size show_shape = img.size();
            plot_results(img, objs, colors, names, show_shape, entry.path().stem());

            cv::imwrite(output_path.string(), img);
            std::cout << "Saved: " << output_path << std::endl;
        }
    }
    return 0;  
    
}

