#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

struct ToFConfig {
    std::string i2c_bus = "/dev/i2c-1";
    uint8_t i2c_address = 0x29;
    std::string xshut_path;  // optional sysfs path to toggle
    int timing_budget_ms = 50;  // measurement timing budget
    int inter_measurement_ms = 60;
    int output_hz = 20;
    bool json_output = true;
};

struct ToFMeasurement {
    uint16_t distance_mm = 0;
    float signal_rate = 0.0f;  // MCPS equivalent (best-effort)
    uint64_t timestamp_ms = 0;
};

class ToFReader {
  public:
    explicit ToFReader(const ToFConfig& cfg);
    ~ToFReader();

    bool init();
    std::optional<ToFMeasurement> read_once();

  private:
    bool reset_sensor();
    int parse_bus_number(const std::string& bus) const;

    ToFConfig config_;
    int bus_number_ = 1;
    bool initialized_ = false;
    std::unique_ptr<class VL53L0X> sensor_;
};

uint64_t monotonic_millis();
bool write_gpio_value(const std::string& path, bool high);

