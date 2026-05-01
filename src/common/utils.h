
#ifndef PLANTSIM_UTILS_H
#define PLANTSIM_UTILS_H

#include <chrono>

long long getCurrentTimeSeconds() {
    // Get the current time point
    auto currentTime = std::chrono::system_clock::now();

    // Convert the time point to a duration since the epoch
    auto durationSinceEpoch = currentTime.time_since_epoch();

    // Convert the duration to seconds
    auto secondsSinceEpoch = std::chrono::duration_cast<std::chrono::seconds>(durationSinceEpoch);

    // Extract the number of seconds
    return secondsSinceEpoch.count();
}

#endif // PLANTSIM_UTILS_H