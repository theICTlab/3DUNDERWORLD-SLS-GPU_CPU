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

/*! Projector projects patterns to reconstruction objects
 */
class Projector
{
protected:
    //! Projector width and height
    size_t width_, height_;
public:
    Projector() = delete;
    //! Initialize projector with height and width
    Projector(size_t width, size_t height):width_{width},height_{height}{}
    virtual ~Projector(){};
    /*! Return the size of project
     * \param w Output width
     * \param h Output height
     */
    void getSize(size_t &w, size_t &h){w = width_; h = height_;}
    //! Get width
    size_t getWidth() const {return width_;}
    //! Get height
    size_t getHeight() const {return height_;}
    //! Get number of pixels = getWidth() * getHeight()
    size_t getNumPixels() const {return width_ * height_;}
    /*! Get required number of frames to distinguish all of the projector pixel.
     *
     * The patterns are designed such that each projector pixel has a unique binary sequence. 
     * One frame can set one bit of the sequences of all projectors. In this application, we
     * want to distinguish on both x and y dimensions. Thus, the required number
     * of frames can be infered from the size of projector: 
     * \f[\left\lceil\log_2{w}\right\rceil+\left\lceil\log_2{h}\right\rceil\f]
     */
    size_t getRequiredNumFrames() const
    {
        return (size_t)std::ceil(std::log2(width_))+std::ceil(std::log2(height_));
    }
};
}
