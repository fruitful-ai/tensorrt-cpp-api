#include "logger.h"

std::string getLogLevelFromEnvironment() {
    const char* envValue = std::getenv("LOG_LEVEL");
    if (envValue) {
        return std::string(envValue);
    } else {
        spdlog::warn("LOG_LEVEL environment variable not set. Using default log level (info).");
        return "info";
    }
}

LogLevel parseLogLevel(const std::string& logLevelStr) {
    if (logLevelStr == "trace") {
        return LogLevel::Trace;
    } else if (logLevelStr == "debug") {
        return LogLevel::Debug;
    } else if (logLevelStr == "info") {
        return LogLevel::Info;
    } else if (logLevelStr == "warn" || logLevelStr == "warning") {
        return LogLevel::Warn;
    } else if (logLevelStr == "err" || logLevelStr == "error") {
        return LogLevel::Error;
    } else if (logLevelStr == "critical") {
        return LogLevel::Critical;
    } else if (logLevelStr == "off") {
        return LogLevel::Off;
    } else {
        spdlog::warn("Unknown log level string: {}. Defaulting to 'info' level.", logLevelStr);
        return LogLevel::Unknown;
    }
}

spdlog::level::level_enum toSpdlogLevel(const std::string& logLevelStr) {
    LogLevel logLevel = parseLogLevel(logLevelStr);
    
    switch (logLevel) {
        case LogLevel::Trace:
            return spdlog::level::trace;
        case LogLevel::Debug:
            return spdlog::level::debug;
        case LogLevel::Info:
            return spdlog::level::info;
        case LogLevel::Warn:
            return spdlog::level::warn;
        case LogLevel::Error:
            return spdlog::level::err;
        case LogLevel::Critical:
            return spdlog::level::critical;
        case LogLevel::Off:
            return spdlog::level::off;
        default:
            spdlog::warn("Unknown log level. Using default log level (info).");
            return spdlog::level::info;
    }
}
