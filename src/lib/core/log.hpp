/*
 * Log stuff;
 */
#pragma once
#include <chrono>
#include <iostream>
#include <ctime>
#define GL_LOG_FILE "sls.log"

namespace LOG
{
    bool writeLog (const char* message, ...);
    bool writeLogErr (const char* message, ...);
    bool restartLog();

    //Log with timer
    static std::chrono::time_point<std::chrono::steady_clock> start;
    bool startTimer(const char* message, ...);
    bool startTimer();
    bool endTimer(const char* message, ...);
    bool endTimer(const char unit='m');

    //display progress bar
    bool progress(const float &prog);




}

