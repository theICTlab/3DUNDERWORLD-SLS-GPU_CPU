/* Copyright (C) 
 * 2016 - Tsing Gu
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 */

#pragma once
#include <iostream>
#include <cmath>
namespace SLS {

/**
 * @brief Patterns that projected to the model
 * This class manages projector
 */
class Projector
{
protected:
    size_t width_, height_;
public:
    Projector() = delete;
    Projector(size_t width, size_t height):width_{width},height_{height}{}
    virtual ~Projector(){};
    void getSize(size_t &w, size_t &h){w = width_; h = height_;}
    size_t getWidth() const {return width_;}
    size_t getHeight() const {return height_;}
    size_t getNumPixels() const {return width_ * height_;}
    size_t getRequiredNumFrames() const
    {
        return (size_t)std::ceil(std::log2(width_))+std::ceil(std::log2(height_));
        //return (size_t )std::ceil(std::log2((float)width_*(float)height_));   // hmmm TODO: Figure out why it doesn't work
    }
};
}
