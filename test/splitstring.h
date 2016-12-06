#include <string>
#include <vector>
#include <iostream>

#ifndef SPLITSTRING_H
#define SPLITSTRING_H
using namespace std;


class splitstring : public string {
    vector<string> flds;
    vector<float> floatArray;
public:
    splitstring(char *s) : string(s) { };
    splitstring(string s) : string(s) { };
    vector<string>& split(char delim, int rep=0);
    vector<float>& splitFloat(char delim, int rep=0);
};
#endif
