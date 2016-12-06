#include "splitstring.h"
// split: receives a char delimiter; returns a vector of strings
// By default ignores repeated delimiters, unless argument rep == 1.
vector<string>& splitstring::split(char delim, int rep) {
    if (!flds.empty()) flds.clear();  // empty vector if necessary
    string work = data();
    string buf = "";
    unsigned i = 0;
    while (i < work.length()) {
        if (work[i] != delim)
            buf += work[i];
        else if (rep == 1) {
            flds.push_back(buf);
            buf = "";
        } else if (buf.length() > 0) {
            flds.push_back(buf);
            buf = "";
        }
        i++;
    }
    if (!buf.empty())
        flds.push_back(buf);
    return flds;
}

vector<float>& splitstring::splitFloat(char delim, int rep) {
    if (!floatArray.empty()) floatArray.clear();  // empty vector if necessary
    string work = data();
    string buf = "";
    unsigned i = 0;
    while (i < work.length()) {
        if (work[i] != delim)
            buf += work[i];
        else if (rep == 1) {
            floatArray.push_back(atof(buf.c_str()));
            buf = "";
        } else if (buf.length() > 0) {
            floatArray.push_back(atof(buf.c_str()));
            buf = "";
        }
        i++;
    }
    if (!buf.empty())
        floatArray.push_back(atof(buf.c_str()));
    return floatArray;
}
