#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
// =================================================

const size_t IMGCOUNT = 10;
const bool ENABLE_RESIZE = true;
const cv::Size RESIZE_TO = cv::Size(1200, 800);

// =================================================

void Resize(Mat &imageInput)
{
    if (ENABLE_RESIZE)
        cv::resize(imageInput, imageInput, RESIZE_TO);
}

void main()
{
    Size image_size;                          // 图像的尺寸
    Size board_size = Size(7, 5);             // 标定板上每行、列的角点数
    vector<Point2f> image_points_buf;         // 缓存每幅图像上检测到的角点
    vector<vector<Point2f>> image_points_seq; // 保存检测到的所有角点
    /*提取角点*/
    char filename[50];
    for (size_t image_num = 0; image_num < IMGCOUNT; image_num++)
    {
        sprintf_s(filename, "../asset/calibration/%d.JPG", image_num);
        Mat imageInput = imread(filename);
        Resize(imageInput);
        if (!findChessboardCorners(imageInput, board_size, image_points_buf))
        {
            cout << "can not find chessboard corners!\n"; //找不到角点
            return;
        }
        else
        {
            Mat view_gray;
            cvtColor(imageInput, view_gray, cv::COLOR_RGB2GRAY);
            /*亚像素精确化*/
            find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5));       // 对粗提取的角点进行精确化
            drawChessboardCorners(view_gray, board_size, image_points_buf, true); // 用于在图片中标记角点
            image_points_seq.push_back(image_points_buf);                         // 保存亚像素角点
            imshow("Camera Calibration", view_gray);                              // 显示图片
            waitKey(500);                                                         // 停半秒
        }
        image_size.width = imageInput.cols;
        image_size.height = imageInput.rows;
        imageInput.release();
    }

    /* 相机标定 */
    vector<vector<Point3f>> object_points; // 保存标定板上角点的三维坐标,为标定函数的第一个参数
    Size square_size = Size(2, 2);         // 实际测量得到的标定板上每个棋盘格的大小，这里其实没测，就假定了一个值，感觉影响不是太大，后面再研究下
    for (int t = 0; t < IMGCOUNT; t++)
    {
        vector<Point3f> tempPointSet;
        for (int i = 0; i < board_size.height; i++)
        {
            for (int j = 0; j < board_size.width; j++)
            {
                Point3f realPoint;
                // 假设标定板放在世界坐标系中z=0的平面上
                realPoint.x = i * square_size.width;
                realPoint.y = j * square_size.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        object_points.push_back(tempPointSet);
    }
    // 内外参数对象
    Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参数矩阵
    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));   // 摄像机的5个畸变系数：k1,k2,p1,p2,k3
    vector<Mat> tvecsMat;                                   // 每幅图像的旋转向量
    vector<Mat> rvecsMat;                                   // 每幅图像的平移向量
    calibrateCamera(object_points,                          // 相机标定
                    image_points_seq,
                    image_size,
                    cameraMatrix,
                    distCoeffs,
                    rvecsMat,
                    tvecsMat, 0);
    /* 用标定的结果矫正图像 */
    for (size_t image_num = 0; image_num <= IMGCOUNT; image_num++)
    {
        sprintf_s(filename, "../asset/calibration/%d.JPG", image_num);
        Mat imageSource = imread(filename);
        Resize(imageSource);
        Mat newimage = imageSource.clone();
        undistort(imageSource, newimage, cameraMatrix, distCoeffs);
        imshow("source", imageSource);                                   // 显示图片
        imshow("drc", newimage);                                         // 显示图片
        sprintf_s(filename, "../asset/calibration/%d_d.JPG", image_num); // 目标文件名
        imwrite(filename, newimage);                                     // 写入文件
        waitKey(500);                                                    // 停半秒
        imageSource.release();
        newimage.release();
    }
    /*保存内参和畸变系数，以便后面直接矫正*/
    ofstream fout("../asset/calibration/caliberation_result.txt"); // 保存标定结果的文件
    fout << "intrinsic matrix:" << endl;
    fout << cameraMatrix << endl
         << endl;
    fout << "distortion parameters:\n";
    fout << distCoeffs << endl
         << endl
         << endl;
    fout.close();

    ///*读取之前标定好的数据直接矫正*/
    //char read[100];
    //double getdata;
    //Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//摄像机内参数矩阵
    //Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));//摄像机的5个畸变系数：k1,k2,p1,p2,k3
    //ifstream fin("caliberation_result.txt");//读取保存标定结果的文件，以供矫正
    //fin >> read;
    //fin.seekg(3, ios::cur);
    //for (size_t j = 0; j < 3; j++)
    //  for (size_t i = 0; i < 3; i++)
    //  {
    //      fin >> getdata;
    //      cameraMatrix.at<float>(j, i) = getdata;
    //      fin >> read;
    //  }
    //fin >> read;
    //fin.seekg(3, ios::cur);
    //for (size_t i = 0; i < 5; i++)
    //{
    //  fin >> getdata;
    //  distCoeffs.at<float>(i) = getdata;
    //  fin >> read;
    //}
    //fin.close();

    //char filename[10];
    //for (size_t image_num = 1; image_num <= IMGCOUNT; image_num++)
    //{
    //  sprintf_s(filename, "%d.bmp", image_num);
    //  Mat imageSource = imread(filename);
    //  Mat newimage = imageSource.clone();
    //  undistort(imageSource, newimage, cameraMatrix, distCoeffs);
    //  imshow("source", imageSource);//显示图片
    //  imshow("drc", newimage);//显示图片
    //  sprintf_s(filename, "%d_d.bmp", image_num);
    //  imwrite(filename, newimage);//显示图片
    //  waitKey(500);//停半秒
    //  imageSource.release();
    //  newimage.release();
    //}
}