#include <librealsense2/rs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace
{
struct Stats
{
    double mean = 0.0;
    double stdev = 0.0;
    double min = 0.0;
    double max = 0.0;
    size_t count = 0;
};

Stats compute_stats(const std::vector<float>& samples)
{
    Stats s;
    s.count = samples.size();
    if (samples.empty())
    {
        return s;
    }

    auto minmax = std::minmax_element(samples.begin(), samples.end());
    s.min = *minmax.first;
    s.max = *minmax.second;

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    s.mean = sum / static_cast<double>(samples.size());

    double variance = 0.0;
    for (float v : samples)
    {
        double diff = v - s.mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(samples.size());
    s.stdev = std::sqrt(variance);
    return s;
}

std::vector<float> sample_depth_patch(const rs2::depth_frame& depth, float roi_ratio, unsigned int stride)
{
    std::vector<float> values;
    const int width = depth.get_width();
    const int height = depth.get_height();
    const int roi_w = static_cast<int>(width * roi_ratio);
    const int roi_h = static_cast<int>(height * roi_ratio);
    const int x0 = (width - roi_w) / 2;
    const int y0 = (height - roi_h) / 2;
    const float depth_unit = depth.get_units();

    for (int y = y0; y < y0 + roi_h; y += static_cast<int>(stride))
    {
        for (int x = x0; x < x0 + roi_w; x += static_cast<int>(stride))
        {
            float d = depth.get_distance(x, y);
            if (d > 0.0f)
            {
                values.push_back(d);
            }
        }
    }

    // Convert to meters once to avoid caller confusion even though get_distance already accounts for units.
    for (float& v : values)
    {
        v *= depth_unit;
    }
    return values;
}

bool evaluate_liveness(const Stats& stats, double min_range_m, double min_stdev_m, size_t min_samples)
{
    if (stats.count < min_samples)
    {
        return false;
    }
    const double range = stats.max - stats.min;
    return (range >= min_range_m) && (stats.stdev >= min_stdev_m);
}

void print_metrics(const Stats& stats, bool alive)
{
    std::cout << "samples=" << stats.count << " min(m)=" << stats.min << " max(m)=" << stats.max << " mean(m)=" << stats.mean
              << " stdev(m)=" << stats.stdev << " -> " << (alive ? "LIVE" : "FLAT") << std::endl;
}

} // namespace

int main()
{
    try
    {
        rs2::context ctx;
        auto list = ctx.query_devices();
        if (list.size() == 0)
        {
            std::cerr << "No RealSense devices connected." << std::endl;
            return 1;
        }

        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

        rs2::pipeline pipe;
        auto profile = pipe.start(cfg);
        std::cout << "Running on device: "
                  << profile.get_device().get_info(RS2_CAMERA_INFO_NAME) << " (SN: "
                  << profile.get_device().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << ")" << std::endl;

        rs2::align align_to_color(RS2_STREAM_COLOR);

        constexpr int warmup_frames = 30;
        for (int i = 0; i < warmup_frames; ++i)
        {
            pipe.wait_for_frames();
        }

        constexpr float roi_ratio = 0.4f;      // central 40% area
        constexpr unsigned int stride = 4;      // subsample to reduce work
        constexpr double min_range_m = 0.04;    // reject flats with <4 cm depth variation
        constexpr double min_stdev_m = 0.01;    // require 1 cm standard deviation
        constexpr size_t min_samples = 250;     // minimum valid depth samples in ROI

        std::cout << "Press Ctrl+C to stop. Capturing..." << std::endl;
        while (true)
        {
            rs2::frameset frames = pipe.wait_for_frames();
            frames = align_to_color.process(frames);
            auto depth = frames.get_depth_frame();
            if (!depth)
            {
                continue;
            }

            auto samples = sample_depth_patch(depth, roi_ratio, stride);
            auto stats = compute_stats(samples);
            bool alive = evaluate_liveness(stats, min_range_m, min_stdev_m, min_samples);
            print_metrics(stats, alive);
        }
    }
    catch (const rs2::error& e)
    {
        std::cerr << "RealSense error: " << e.get_failed_function() << ": " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}
