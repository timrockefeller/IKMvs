#include <iostream>
#include <IKMvs/ImageQueue.h>
#include <opencv2/opencv.hpp>

using namespace cv;
int main(int argc, char** argv )
{
    std::cout<<"Well you made it! \n";
    Mat image;
    image = imread( "../asset/awesomeface.png", 1 );

    if ( !image.data )
    {
        std::cout<<"No image data \n";
        waitKey(0);
        return -1;
    }
    namedWindow("GoodJob", WINDOW_AUTOSIZE );
    imshow("GoodJob", image);

    waitKey(0);

    return 0;
}