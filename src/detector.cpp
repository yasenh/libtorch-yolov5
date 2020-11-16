#include "detector.h"


Detector::Detector(const std::string& model_path, const torch::DeviceType& device_type) : device_(device_type) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        std::exit(EXIT_FAILURE);
    }

    half_ = (device_ != torch::kCPU);
    module_.to(device_);

    if (half_) {
        module_.to(torch::kHalf);
    }

    module_.eval();
}


std::vector<std::vector<Detection>>
Detector::Run(const cv::Mat& img, float conf_threshold, float iou_threshold) {
    torch::NoGradGuard no_grad;
    std::cout << "----------New Frame----------" << std::endl;

    // TODO: check_img_size()

    /*** Pre-process ***/

    auto start = std::chrono::high_resolution_clock::now();

    // keep the original image for visualization purpose
    cv::Mat img_input = img.clone();

    std::vector<float> pad_info = LetterboxImage(img_input, img_input, cv::Size(640, 640));
    const float pad_w = pad_info[0];
    const float pad_h = pad_info[1];
    const float scale = pad_info[2];

    cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB);  // BGR -> RGB
    img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(device_);

    tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)

    if (half_) {
        tensor_img = tensor_img.to(torch::kHalf);
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "pre-process takes : " << duration.count() << " ms" << std::endl;

    /*** Inference ***/
    // TODO: add synchronize point
    start = std::chrono::high_resolution_clock::now();

    // inference
    torch::jit::IValue output = module_.forward(inputs);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "inference takes : " << duration.count() << " ms" << std::endl;

    /*** Post-process ***/

    start = std::chrono::high_resolution_clock::now();
    auto detections = output.toTuple()->elements()[0].toTensor();

    // result: n * 7
    // batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
    auto result = PostProcessing(detections, pad_w, pad_h, scale, img.size(), conf_threshold, iou_threshold);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "post-process takes : " << duration.count() << " ms" << std::endl;

    return result;
}


std::vector<float> Detector::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
    int left = (static_cast<int>(out_w)- mid_w) / 2;
    int right = (static_cast<int>(out_w)- mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}


std::vector<std::vector<Detection>> Detector::PostProcessing(const torch::Tensor& detections,
                                                             float pad_w, float pad_h, float scale, const cv::Size& img_shape,
                                                             float conf_thres, float iou_thres) {
    constexpr int item_attr_size = 5;
    int batch_size = detections.size(0);
    // number of classes, e.g. 80 for coco dataset
    auto num_classes = detections.size(2) - item_attr_size;

    // get candidates which object confidence > threshold
    auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

    std::vector<std::vector<Detection>> output;
    output.reserve(batch_size);

    // iterating all images in the batch
    for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        // apply constrains to get filtered detections for current image
        auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes + item_attr_size});

        // if none detections remain then skip and start to process next image
        if (0 == det.size(0)) {
            continue;
        }

        // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
        det.slice(1, item_attr_size, item_attr_size + num_classes) *= det.select(1, 4).unsqueeze(1);

        // box (center x, center y, width, height) to (x1, y1, x2, y2)
        torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));

        // [best class only] get the max classes score at each result (e.g. elements 5-84)
        std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

        // class score
        auto max_conf_score = std::get<0>(max_classes);
        // index
        auto max_conf_index = std::get<1>(max_classes);

        max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
        max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

        // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
        det = torch::cat({box.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

        // for batched NMS
        constexpr int max_wh = 4096;
        auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
        auto offset_box = det.slice(1, 0, 4) + c;

        std::vector<cv::Rect> offset_box_vec;
        std::vector<float> score_vec;

        // copy data back to cpu
        auto offset_boxes_cpu = offset_box.cpu();
        auto det_cpu = det.cpu();
        const auto& det_cpu_array = det_cpu.accessor<float, 2>();

        // use accessor to access tensor elements efficiently
        Tensor2Detection(offset_boxes_cpu.accessor<float,2>(), det_cpu_array, offset_box_vec, score_vec);

        // run NMS
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);

        std::vector<Detection> det_vec;
        for (int index : nms_indices) {
            Detection t;
            const auto& b = det_cpu_array[index];
            t.bbox =
                    cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]),
                             cv::Point(b[Det::br_x], b[Det::br_y]));
            t.score = det_cpu_array[index][Det::score];
            t.class_idx = det_cpu_array[index][Det::class_idx];
            det_vec.emplace_back(t);
        }

        ScaleCoordinates(det_vec, pad_w, pad_h, scale, img_shape);

        // save final detection for the current image
        output.emplace_back(det_vec);
    } // end of batch iterating

    return output;
}


void Detector::ScaleCoordinates(std::vector<Detection>& data,float pad_w, float pad_h,
                                float scale, const cv::Size& img_shape) {
    auto clip = [](float n, float lower, float upper) {
        return std::max(lower, std::min(n, upper));
    };

    std::vector<Detection> detections;
    for (auto & i : data) {
        float x1 = (i.bbox.tl().x - pad_w)/scale;  // x padding
        float y1 = (i.bbox.tl().y - pad_h)/scale;  // y padding
        float x2 = (i.bbox.br().x - pad_w)/scale;  // x padding
        float y2 = (i.bbox.br().y - pad_h)/scale;  // y padding

        x1 = clip(x1, 0, img_shape.width);
        y1 = clip(y1, 0, img_shape.height);
        x2 = clip(x2, 0, img_shape.width);
        y2 = clip(y2, 0, img_shape.height);

        i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
    }
}


torch::Tensor Detector::xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::zeros_like(x);
    // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
    y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
    y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
    y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
    y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
    return y;
}


void Detector::Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                const at::TensorAccessor<float, 2>& det,
                                std::vector<cv::Rect>& offset_box_vec,
                                std::vector<float>& score_vec) {

    for (int i = 0; i < offset_boxes.size(0) ; i++) {
        offset_box_vec.emplace_back(
                cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
                         cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y]))
        );
        score_vec.emplace_back(det[i][Det::score]);
    }
}