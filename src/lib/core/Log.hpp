#pragma once
#include <chrono>
#include <iostream>
#include <ctime>
#define GL_LOG_FILE "sls.log"   // !< Define log file

/*! System wide log to output message to terminal and file
 */

namespace LOG
{
    // Printf like log file 

    //! Write message to log file only
    bool writeLog (const char* message, ...);

    //! Write message to log file and stderr
    bool writeLogErr (const char* message, ...);

    //! Clean log file
    bool restartLog();

    //Log with timer
    static std::chrono::time_point<std::chrono::steady_clock> start;

    /*
     * Timer functions are used to profile performance of code blocks;
     * ```
     * startTimer("Profiling foo()");
     * foo();
     * endTimer('s');
     * ```
     * 
     */
    //! Start timer with a log
    bool startTimer(const char* message, ...);

    //! Start timer
    bool startTimer();

    //! End timer with a message, the unit is seconds
    bool endTimer(const char* message, ...);

    /*! End timer without message. 
     * \param unit 'm': minutes, 's' seconds, 'n' nanoseconds
     */
    bool endTimer(const char unit='m');

    //display progress bar
    bool progress(const float &prog);


}
