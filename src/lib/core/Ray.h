#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
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
    glm::vec3 p_0 (r1.origin);
    glm::vec3 q_0 (r2.origin);
    glm::vec3 u (r1.dir);
    glm::vec3 v (r2.dir);


    auto w_0 = p_0 - q_0;


    //auto s_c = (dot(u,v)*dot(v,w_0)-dot(u,w_0))/(1-dot(u,v)*dot(u,v));
    //auto t_c = (dot(v,w_0)-dot(u,v)*dot(u, w_0))/(1-dot(u, v)*dot(u, v));
    //
    // alternative
    auto s_c = (dot(u,w_0)*dot(v,v)-dot(u,v)*dot(v,w_0))/
        (dot(u,u)*dot(v,v)-dot(u,v)*dot(u,v));
    auto t_c = (dot(u,v)*dot(u,w_0)-dot(v,w_0)*dot(u,u))/
        (dot(u,u)*dot(v,v)-dot(u,v)*dot(u,v));
    

    //cout<<"s_c="<<s_c<<endl;

    auto p_s = q_0 + t_c * v;
    auto q_c = p_0 + s_c * u;
    dist = distance(p_s, q_c);
    return vec4((p_s + q_c)/2.0f, 1.0);
}
}
