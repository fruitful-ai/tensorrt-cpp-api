#pragma once

#include <string>
#include <spdlog/spdlog.h>

enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
    Off,
    Unknown
};

std::string getLogLevelFromEnvironment();
LogLevel parseLogLevel(const std::string& logLevelStr);
spdlog::level::level_enum toSpdlogLevel(const std::string& logLevelStr);
