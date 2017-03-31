#pragma once
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "Log.hpp"
using namespace glm;
namespace SLS
{
struct Ray
{
    vec4 origin;
    vec4 dir;
    vec3 color;     // RGB Color
};

/**
 *! Get mid point of the segment perpendicular to both rays, i.e. intersection point.
 *
 * /param r1 Input of ray 1
 * /param r2 Input of ray 2
 * /param dist distance between two rays
 *
 * /return Mid point
 */
inline glm::vec4 midPoint( const Ray &r1, const Ray &r2, float &dist)
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
        // Parallel rays
        dist = -1.0;
        return vec4(0.0);
    }

    float s =  (v1_dot_v2/denom) * v12_dot_v2 - (v2_dot_v2/denom) * v12_dot_v1;
    float t = -(v1_dot_v2/denom) * v12_dot_v1 + (v1_dot_v1/denom) * v12_dot_v2;
    dist = glm::length(p1+s*v1-p2-t*v2);
    return vec4((p1+s*v1+p2+t*v2)/2.0f, 1.0);

}
}
