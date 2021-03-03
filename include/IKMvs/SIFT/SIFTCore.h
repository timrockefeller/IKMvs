#pragma once
#include <IKit/STL/Singleton.h>
#include <opencv2/opencv.hpp>
namespace KTKR::MVS
{

    class SIFTCore : KTKR::Singleton<SIFTCore>
    {
        const static size_t GAUSSKERN = 3.5;


    };

} // namespace KTKR::MVS
