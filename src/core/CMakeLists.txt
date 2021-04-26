IK_AddTarget(
    MODE STATIC
    INC 
        "${PROJECT_SOURCE_DIR}/include"
        ${CERES_INCLUDE_DIRS}
    LIB
        ${OpenCV_LIBS}
        KTKR::IKit_core
        ${CERES_LIBRARIES}
)