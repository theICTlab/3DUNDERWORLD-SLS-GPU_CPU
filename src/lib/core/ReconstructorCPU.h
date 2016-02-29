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
#include "Reconstructor.h"
#include "log.hpp"
#include <new>
namespace SLS
{

/**
 * @brief CPU reconstructor
 */
class ReconstructorCPU: public Reconstructor
{
private:
    std::vector<std::vector<std::vector<size_t>>> buckets_;
    void initBuckets();
    void generateBuckets();
public:
    ReconstructorCPU(const size_t projX, const size_t projY): 
        Reconstructor()
        //projector_{new Projector(projX, projY)}
    {
        try{
            projector_ = new Projector(projX, projY);
        }
        catch(std::bad_alloc& ba)
        {
            LOG::writeLogErr("Bad allocation\n");
            exit(0);
        }
    }
    ~ReconstructorCPU() override;
    //Interfaces
    void renconstruct() override;
    void addCamera(Camera *cam) override;
};
} // namespace SLS
