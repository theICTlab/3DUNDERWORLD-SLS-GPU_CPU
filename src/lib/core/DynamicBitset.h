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
#include <string>
#include <vector>
#include <bitset>
#include <stdexcept>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <glm/glm.hpp>
using std::ostream;

namespace SLS
{
class Dynamic_Bitset{
private:
    std::vector<unsigned char> bits; //!< Bytes used to store bits
protected:
    const size_t BITS_PER_BYTE;

    /* Set or clear a bit in uchar
     * position should be within BIT_PER_BYTE
     * http://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit-in-c-c
     */

    /*! Set bit of a char to 1
     *
     * \param ch Char to operate
     * \param pos Position within the char
     */
    void setUChar(unsigned char& ch, const size_t &pos) { ch |= 1<<pos; }
    /*! Set bit of a char to 0
     *
     * \param ch Char to operate
     * \param pos Position within the char
     */
    void clearUChar(unsigned char& ch, const size_t &pos) { ch &= ~(1<<pos);}

    /*! Get bit of within a char
     *
     * \param ch Char to query
     * \param pos position of bit in char
     *
     * \return Ture if 1; otherwise, 0.
     */
    bool getUChar(const unsigned char& ch, const size_t &pos)const {return (ch>>pos)&1;}
public:
    /*! Init an empty bitset
     */
    Dynamic_Bitset(): BITS_PER_BYTE{8}{}

    /*! Init an empty bitset with given length
     *
     * \param sz number of bits 
     */
    explicit Dynamic_Bitset(const size_t &sz):BITS_PER_BYTE{8}{resize(sz);}
    Dynamic_Bitset(const size_t &sz, unsigned char* b):BITS_PER_BYTE{8}
    {
        bits.resize(sz);
        std::cout<<sz<<std::endl;
        std::cout<<bits.size()<<std::endl;
        assert(sz == bits.size());
        memcpy( &bits[0], b, sz);
    }

    /*!  Get number of bits
     *
     * \return number of bits
     */
    virtual size_t size() const {return bits.size()*BITS_PER_BYTE;}

    /*! Resize bit array length, will set bit array to 0
     *
     * \param sz number of bits
     */
    virtual void resize(const size_t &sz) {
        bits.resize((sz+BITS_PER_BYTE-1)/ BITS_PER_BYTE, 0);
        memset(&(bits[0]), 0, sizeof(unsigned char)*bits.size());
    }

    /*! Set a bit to 1
     *
     * \param pos 0-based position of bit
     */
    void setBit(const size_t &pos){    //Set the bit to one
        if (pos > size())
            throw std::out_of_range("bit access out of range");
        setUChar(bits[pos/BITS_PER_BYTE], pos%BITS_PER_BYTE);
    }

    /*! Set a bit to 0
     *
     * \param pos 0-based position of bit
     */
    void clearBit(const size_t &pos){  //Set the bit to zero
        if (pos > size())
            throw std::out_of_range("bit access out of range");
        clearUChar(bits[pos/BITS_PER_BYTE], pos%BITS_PER_BYTE);
    }

    /*! Get the value of bit
     *
     * \param pos 0-based position of bit
     *
     * \return Ture if the bit is 1; otherwise, false.
     */
    bool getBit(const size_t &pos) const {
        if (pos > size())
            throw std::out_of_range("bit access out of range");
        return getUChar(bits[pos/BITS_PER_BYTE], pos%BITS_PER_BYTE);
    }

    /*! Convert bit array to unsigned int
     *
     * \return representation of bit array in unsigned int
     */
    unsigned int to_uint() const
    {
        if ( bits.size() > sizeof(unsigned int) )
            throw std::overflow_error("Bit array is too long for uint");
        unsigned int res=0;
        for (size_t i=0; i<bits.size(); i++)    //This size is number of bytes
            res += ((unsigned int)bits[i])<<(i*BITS_PER_BYTE);
        return res;
    }

    /*! Convert bitarray to gray code
     */
    glm::uvec2 to_uint_gray () const
    {
        unsigned num = to_uint(); //Gray code

        //TODO: Hack, fix the constant bit size
        //Extract lower 10
        unsigned yDec = num & 0x3FFU;
        //Extract higher 10
        unsigned xDec = num >> 10;

        // Convert lower and higher to reflected dec
        for (unsigned bit=1U<<31; bit>1; bit>>=1) {
            if (xDec & bit) xDec ^= bit >> 1;
        }

        for (unsigned bit=1U<<31; bit>1; bit>>=1) {
            if (yDec & bit) yDec ^= bit >> 1;
        }
        return glm::uvec2(xDec, yDec);
    }
    bool writeToPGM(std::string fileName, const size_t &w, const size_t &h, bool transpose=false);

    //Friend operators
    inline friend std::ostream& operator<<(std::ostream& os, const Dynamic_Bitset& db);
};

std::ostream& operator<<(std::ostream& os, const Dynamic_Bitset& db)
{
    for (std::vector<unsigned char>::const_reverse_iterator rit = db.bits.rbegin();
            rit != db.bits.rend(); rit++)
        os << (std::bitset<8>)(*rit);
    return os;
}
};
