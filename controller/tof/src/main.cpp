#include "tof_reader.hpp"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <optional>
#include <time.h>

namespace {
volatile std::sig_atomic_t g_should_exit = 0;

void signal_handler(int) {
    g_should_exit = 1;
}

void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [--bus /dev/i2c-1] [--addr 0x29]"
              << " [--xshut /sys/class/gpio/gpio4/value] [--hz 20] [--plain]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    ToFConfig cfg;

    static struct option long_opts[] = {
        {"bus", required_argument, nullptr, 'b'},
        {"addr", required_argument, nullptr, 'a'},
        {"xshut", required_argument, nullptr, 'x'},
        {"hz", required_argument, nullptr, 'f'},
        {"plain", no_argument, nullptr, 'p'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'b':
                cfg.i2c_bus = optarg;
                break;
            case 'a':
                cfg.i2c_address = static_cast<uint8_t>(std::strtoul(optarg, nullptr, 0));
                break;
            case 'x':
                cfg.xshut_path = optarg;
                break;
            case 'f':
                cfg.output_hz = std::max(1, std::atoi(optarg));
                break;
            case 'p':
                cfg.json_output = false;
                break;
            case 'h':
            default:
                usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    ToFReader reader(cfg);
    if (!reader.init()) {
        std::cerr << "Failed to initialize VL53L0X" << std::endl;
        return 2;
    }

    const double interval_ms = 1000.0 / cfg.output_hz;
    uint64_t next_deadline = monotonic_millis();

    while (!g_should_exit) {
        auto measurement = reader.read_once();
        if (measurement) {
            if (cfg.json_output) {
                std::cout << "{\"distance_mm\":" << measurement->distance_mm
                          << ",\"signal\":" << measurement->signal_rate
                          << ",\"timestamp_ms\":" << measurement->timestamp_ms
                          << "}" << std::endl;
            } else {
                std::cout << measurement->distance_mm << std::endl;
            }
        }
        next_deadline += static_cast<uint64_t>(interval_ms);
        uint64_t now = monotonic_millis();
        if (next_deadline > now) {
            uint64_t sleep_ms = next_deadline - now;
            struct timespec ts {
                .tv_sec = static_cast<time_t>(sleep_ms / 1000),
                .tv_nsec = static_cast<long>((sleep_ms % 1000) * 1'000'000)
            };
            nanosleep(&ts, nullptr);
        } else {
            next_deadline = now;
        }
    }

    return 0;
}
