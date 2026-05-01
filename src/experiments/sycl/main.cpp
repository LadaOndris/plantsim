#include <iostream>
#include <sycl/sycl.hpp>

// Define a kernel name for the single_task
class hello_world_kernel; 

int main() {
    for (auto platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices())
        {
            std::cout << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()
                      << std::endl;
        }
    }

    return 0;
}