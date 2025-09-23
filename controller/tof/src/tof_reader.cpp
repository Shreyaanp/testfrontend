#include "tof_reader.hpp"

#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <vl53lXx/vl53l0x.hpp>

uint64_t monotonic_millis() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

bool write_gpio_value(const std::string& path, bool high) {
    if (path.empty()) {
        return true;
    }
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Failed to open GPIO path: " << path << std::endl;
        return false;
    }
    ofs << (high ? "1" : "0");
    return ofs.good();
}

ToFReader::ToFReader(const ToFConfig& cfg) : config_(cfg) {}

ToFReader::~ToFReader() = default;

int ToFReader::parse_bus_number(const std::string& bus) const {
    auto pos = bus.rfind("i2c-");
    if (pos != std::string::npos) {
        try {
            return std::stoi(bus.substr(pos + 4));
        } catch (...) {
        }
    }
    // Fallback: try to parse trailing digits
    std::string digits;
    for (auto it = bus.rbegin(); it != bus.rend(); ++it) {
        if (std::isdigit(*it)) {
            digits.insert(digits.begin(), *it);
        } else if (!digits.empty()) {
            break;
        }
    }
    if (!digits.empty()) {
        try {
            return std::stoi(digits);
        } catch (...) {
        }
    }
    return 1;
}

bool ToFReader::reset_sensor() {
    if (config_.xshut_path.empty()) {
        return true;
    }
    if (!write_gpio_value(config_.xshut_path, false)) {
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (!write_gpio_value(config_.xshut_path, true)) {
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return true;
}

bool ToFReader::init() {
    bus_number_ = parse_bus_number(config_.i2c_bus);

    if (!reset_sensor()) {
        return false;
    }

    sensor_ = std::make_unique<VL53L0X>(static_cast<uint8_t>(bus_number_), config_.i2c_address, -1, true);
    sensor_->setTimeout(100);

    if (!sensor_->init()) {
        std::cerr << "VL53L0X init failed" << std::endl;
        sensor_.reset();
        return false;
    }

    if (!sensor_->setMeasurementTimingBudget(static_cast<uint32_t>(config_.timing_budget_ms) * 1000U)) {
        std::cerr << "Failed to set measurement timing budget" << std::endl;
    }

    sensor_->startContinuous(static_cast<uint32_t>(config_.inter_measurement_ms));
    initialized_ = true;
    return true;
}

std::optional<ToFMeasurement> ToFReader::read_once() {
    if (!initialized_ || !sensor_) {
        return std::nullopt;
    }

    uint16_t distance = sensor_->readRangeContinuousMillimeters(false);
    if (sensor_->timeoutOccurred()) {
        std::cerr << "VL53L0X measurement timeout" << std::endl;
        return std::nullopt;
    }

    if (distance == 0 || distance > 4000) {
        return std::nullopt;
    }

    ToFMeasurement measurement;
    measurement.distance_mm = distance;
    measurement.signal_rate = 0.0f;
    measurement.timestamp_ms = monotonic_millis();
    return measurement;
}
