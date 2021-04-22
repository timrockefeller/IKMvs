#pragma once
namespace KTKR::MVS
{

    class SfM
    {
    private:
        void extractFeatures();

    public:
        void runSfM();
    };
} // namespace KTKR::MVS