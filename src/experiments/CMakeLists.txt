list(FIND BACKEND_LIST "SYCL" _has_sycl)
if(NOT _has_sycl EQUAL -1)
    add_subdirectory(sycl)
endif()
