#pragma once

enum Det {
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};

constexpr float kConfThreshold = 0.4;
constexpr float kIouThreshold = 0.6;