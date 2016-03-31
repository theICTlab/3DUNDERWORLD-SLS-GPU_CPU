#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include "log.hpp"
using namespace glm;
namespace SLS
{
struct Ray
{
    vec4 origin;
    vec4 dir;
};

/**
 * @brief Get mid point of two rays.
 *
 * @param r1 Input of ray 1
 * @param r2 Input of ray 2
 * @param dist distance between two rays
 *
 * @return Mid point
 */
inline glm::vec4 midPoint(const Ray &r1, const Ray &r2, float &dist)
{
    glm::vec3 a (r1.origin);
    glm::vec3 b (r1.dir);
    glm::vec3 c (r2.origin);
    glm::vec3 d (r2.dir);




    auto s = ( dot(b,d)*( dot(a,d) - dot(b,c)) - dot(a,d)*dot(c,d)) / (dot (b,d)*dot(b,d)-1.0f);
    auto t = ( dot(b,d)*( dot(c,d) - dot(a,d)) - dot(b,c)*dot(a,b)) / (dot(b,d)*dot(b,d)-1.0f);
    

    auto p_1 = a+s*b;
    auto q_1 = c+t*d;

    dist = distance(p_1, q_1);
    return vec4((p_1+q_1)/2.0f, 1.0f);

}
}
