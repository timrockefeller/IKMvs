/**
 * Calibration Target Generator
 * 
 * 
 * Easy tool for camera calibration without printing a extra image board.
 * Optifined with global error from double. You can edit parameters commented below.
 * 
 * Kitekii 2021 copyright.
 */
#include <iostream>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

// 显示器点距 (mm)
// screen model: 276E8FJAB
const double PIX_DISTANCE = 0.2331;

constexpr int castToPixel(const double worldPos, const double pixDistance = PIX_DISTANCE) noexcept
{
    return static_cast<int>(worldPos / pixDistance);
}

void main()
{

    // =================================================

    const int n_cols = 7;            // 角点行数
    const int n_rows = 5;            // 角点列数
    const double block_length = 20; // 世界坐标边距 (mm)
    const int border_pix = 60;       // 生成标定图的边距(pix)

    // =================================================

    const double col_length = block_length * (n_cols + 1);
    const double row_length = block_length * (n_rows + 1);
    const int col_pix = castToPixel(col_length);
    const int row_pix = castToPixel(row_length);
    Mat map = Mat(border_pix * 2 + row_pix, border_pix * 2 + col_pix, CV_8UC3, Scalar::all(255));
    for (int i = 0; i < row_pix; i++)
        for (int j = 0; j < col_pix; j++)
            if (!(int(1.0 * i / row_pix * n_rows) % 2 == 0 && int(1.0 * j / col_pix * n_cols) % 2 != 0 || int(1.0 * i / row_pix * n_rows) % 2 != 0 && int(1.0 * j / col_pix * n_cols) % 2 == 0))
                map.at<Vec3b>(border_pix + i, border_pix + j)[0] = map.at<Vec3b>(border_pix + i, border_pix + j)[1] = map.at<Vec3b>(border_pix + i, border_pix + j)[2] = 0;
    imshow("map", map);
    waitKey(0);
    return;
}