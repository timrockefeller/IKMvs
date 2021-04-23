IK_AddTarget(
    MODE STATIC
    INC "${PROJECT_SOURCE_DIR}/include"
    LIB
        ${OpenCV_LIBS}
        KTKR::IKit_core
)