#pragma once
#include <opencv2/opencv.hpp>
#include <Ikit/STL/Singleton.h>
#include <fstream>
#include <string>
namespace KTKR::MVS
{
    class CalibrationManager : public KTKR::Singleton<CalibrationManager>
    {
    private:
        cv::Size board_size;
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;

    public:
        CalibrationManager()
        {
            camera_matrix = cv::Mat(3, 3, CV_64FC1, cv::Scalar::all(0));
            dist_coeffs = cv::Mat(1, 5, CV_64FC1, cv::Scalar::all(0));
        }

        bool SaveCalibrationParameters(std::string filepath) noexcept
        {
            std::ofstream fout(filepath);
            fout << "intrinsic matrix:" << std::endl;
            fout << camera_matrix << std::endl
                 << std::endl
                 << "distortion parameters:" << std::endl
                 << dist_coeffs << std::endl;
            fout.close();
        }
        bool ReadCalibrationParameters(const std::string &filepath) noexcept
        {
            std::ifstream fin(filepath);
            if (fin)
            {
                char read[100];
                double getdata;
                fin >> read >> read;
                fin.seekg(3, std::ios::cur);
                for (int j = 0; j < 3; j++)
                    for (int i = 0; i < 3; i++)
                    {
                        fin >> getdata;
                        camera_matrix.at<double>(j, i) = getdata;
                        fin >> read;
                    }
                fin >> read >> read;
                fin.seekg(3, std::ios::cur);
                for (int i = 0; i < 5; i++)
                {
                    fin >> getdata;
                    dist_coeffs.at<double>(i) = getdata;
                    fin >> read;
                }
                fin.close();
                return true;
            }
            return false;
        }
        void SetBoardSize(cv::Size boardSize = cv::Size(7, 5)) { board_size = std::move(boardSize); }
        cv::Mat GetCameraMatrix() noexcept { return camera_matrix; }
        cv::Mat GetDistortion() noexcept { return dist_coeffs; }
    };

} // namespace KTKR::MVS
