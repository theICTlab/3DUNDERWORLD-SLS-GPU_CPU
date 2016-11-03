#include "Log.hpp"
#include <cstdarg>
#include <cstdio>
#include <ctime>

namespace LOG
{
    bool writeLog (const char* message, ...)
    {
        va_list argptr;
        FILE* file = fopen (GL_LOG_FILE, "a");
        if (!file) {
            fprintf (
                    stderr,
                    "ERROR: could not open GL_LOG_FILE %s file for appending\n",
                    GL_LOG_FILE
                    );
            return false;
        }
        va_start (argptr, message);
        vfprintf (file, message, argptr);
        va_end (argptr);
        fclose (file);
        return true;
    }
    bool writeLogErr(const char* message, ...)
    {
        va_list argptr;
        FILE* file = fopen (GL_LOG_FILE, "a");
        if (!file) {
            fprintf (
                    stderr,
                    "ERROR: could not open GL_LOG_FILE %s file for appending\n",
                    GL_LOG_FILE
                    );
            return false;
        }
        va_start (argptr, message);
        vfprintf (file, message, argptr);
        va_end (argptr);
        va_start (argptr, message);
        vfprintf (stderr, message, argptr);
        va_end (argptr);
        fclose (file);
        return true;
    }
    bool restartLog()
    {
        FILE* file = fopen (GL_LOG_FILE, "w");
        if (!file) {
            fprintf (
                    stderr,
                    "ERROR: could not open GL_LOG_FILE log file %s for writing\n",
                    GL_LOG_FILE
                    );
            return false;
        }
        time_t now = time (NULL);
        char* date = ctime (&now);
        fprintf (file, "GL_LOG_FILE log. local time %s\n", date);
        fclose (file);
        return true;
    }
    bool startTimer(const char* message, ...)
    {
        va_list argptr;
        FILE* file = fopen (GL_LOG_FILE, "a");
        if (!file) {
            fprintf (
                    stderr,
                    "ERROR: could not open GL_LOG_FILE %s file for appending\n",
                    GL_LOG_FILE
                    );
            return false;
        }
        va_start (argptr, message);
        vfprintf (file, message, argptr);
        va_end (argptr);
        fclose (file);

        start = std::chrono::steady_clock::now();
        return true;
    }
    bool startTimer()
    {
        start = std::chrono::steady_clock::now();
        return true;
    }
    bool endTimer(const char* message, ...)
    {
        va_list argptr;
        FILE* file = fopen (GL_LOG_FILE, "a");
        if (!file) {
            fprintf (
                    stderr,
                    "ERROR: could not open GL_LOG_FILE %s file for appending\n",
                    GL_LOG_FILE
                    );
            return false;
        }
        va_start (argptr, message);
        vfprintf (file, message, argptr);
        va_end (argptr);
        fclose(file);
        return endTimer('s');
    }
    bool endTimer(const char unit)
    {
        auto diff = std::chrono::steady_clock::now()-start;
        switch (unit){
            case 'm':
                return writeLog(" %fms\n",std::chrono::duration <double, std::milli> (diff).count());
            case 'n':
                return writeLog(" %fns\n",std::chrono::duration <double, std::nano> (diff).count());
            default:
                return writeLog(" %fs\n",std::chrono::duration <double, std::ratio<1,1>> (diff).count());
        }
        return true;
    }

    //Prograss bar is slow. Use it while you are loading some big files in few iterations
    bool progress(const float &prog)
    {
        if (prog > 1.0) return false;
        int barWidth = 70;
        std::cout << "[";
        int pos = barWidth * prog;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(prog * 100.0) << " %\r";
        std::cout.flush();
        return true;
    }
}
