#pragma once
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
//#include <glm/gtx/string_cast.hpp>
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
inline glm::vec4 midPointBkp( const Ray &r1, const Ray &r2, float &dist)
{
    glm::vec3 v1 (r1.dir);
    glm::vec3 v2 (r2.dir);
    glm::vec3 p1 (r1.origin);
    glm::vec3 p2 (r2.origin);
    glm::vec3 v12 = p1-p2;
    float v1_dot_v1 = dot(v1, v1);
    float v2_dot_v2 = dot(v2, v2);
    float v1_dot_v2 = dot(v1, v2); 
    float v12_dot_v1 = dot(v12, v1);
    float v12_dot_v2 = dot(v12, v2);

    float denom = v1_dot_v1 * v2_dot_v2 - v1_dot_v2 * v1_dot_v2;
    if (glm::abs(denom) < 0.1)
    {
            dist = -1.0;
            return vec4(0.0);
    }

    float s =  (v1_dot_v2/denom) * v12_dot_v2 - (v2_dot_v2/denom) * v12_dot_v1;
    float t = -(v1_dot_v2/denom) * v12_dot_v1 + (v1_dot_v1/denom) * v12_dot_v2;
    dist = glm::length(p1+s*v1-p2-t*v2);
    return vec4((p1+s*v1+p2+t*v2)/2.0f, 1.0);

}
inline glm::vec4 midPointPaper(const Ray &r1, const Ray &r2, float &dist)
{
    glm::vec3 p(r1.origin);
    glm::vec3 q(r2.origin);
    glm::vec3 u(r1.dir);
    glm::vec3 v(r2.dir);
    auto w = p-q;
    auto s = (dot(w,u)*dot(v,v)-dot(u,v)*dot(w,v))/(dot(v,u)*dot(v,u) - dot(v,v)*dot(u,u));
    auto t = (dot(v,u)*dot(w,u)-dot(u,u)*dot(w,v))/(dot(v,u)*dot(v,u)-dot(v,v)*dot(u,u));
    return vec4( ((p+s*u)+(q+t*v))/2.0f, 1.0);
}
}
