//------------------------------------------------------------------------------------------------------------
//* Copyright © 2010-2015 Immersive and Creative Technologies Lab, Cyprus University of Technology           *
//* Link: http://www.theICTlab.org                                                                           *
//* Software developer(s): Kyriakos Herakleous                                                               *
//* Researcher(s): Kyriakos Herakleous, Charalambos Poullis                                                  *
//*                                                                                                          *
//* License: Check the file License.md                                                                       *
//------------------------------------------------------------------------------------------------------------


//random number generator
#ifndef __RNG_H__
#define __RNG_H__

#include <math.h>
#include <stdlib.h>

#define IA   16807
#define IM   2147483647
#define IQ   127773
#define IR   2836
#define MASK 123459876
#define NTAB 32
#define EPS  1.2e-7
#define RNMX (1.0-EPS)


class RNG
{
public:
   RNG(unsigned long long _seed = 7564231ULL) 
   {
      seed      = _seed;
      mult      = 62089911ULL;
      llong_max = 4294967295ULL;
      double_max = 4294967295.0f;
   }
   double operator()();
	
   double uniformRandCLib(double _min,double _max);
   double fastUniformRand(double _min, double _max);
   double uniformRand(double _min, double _max);
   double normalRand(void);
   double normalRand(double mu, double sigma);


   unsigned long long seed;
   unsigned long long mult;
   unsigned long long llong_max;
   double double_max;
};

inline double RNG::operator()()
{
   seed = mult * seed;
   return double(seed % llong_max) / double_max;
}

/*! ************************************************************************* 

    Function:
        CRandomNumberGenerator::FastUniformRand()

    Description:
        Generates a random number between min and max inclusively using a
        uniform random deviate.

    Parameters:
        min - the minimum number that can be generated
        max - the maximum number that can be generated

    Return Value:
        A random number between min and max inclusively.

    Notes:
        This algoirthm is adapted from:

            Press, W, Teukolsky, S, et al., "Numerical Recipes in C", 2nd Ed,
            pp 283-285, Cambridge Press, 1996.

*****************************************************************************/


inline double RNG::fastUniformRand(double _min, double _max) {

    int k;
    double ans;

    seed ^= MASK;                     // XORing with Mask allows use of zero for idnum

    k = seed / IQ;

    seed = IA*(seed-k*IQ) - IR*k;   // compute idnum=(IA*idum) % IM iwth overflows by Schrage's method
    if( seed < 0) {
        seed += IM;
    }

    ans = (1.0f/IM)*seed;             // convert idum to a doubleing result

    seed ^= MASK;                     // unmask before return

    ans = ans*(_max-_min) + _min;            // shift number into desired range

    return ans;
}


/*! ************************************************************************* 

    Function:
        CRandomNumberGenerator::UniformRandCLib()

    Description:
        Generates a random number between min and max inclusively using the
        uniform random deviate from the C standard library.  It is beter to
        use the function UniformRand().  For a discussion see:

            Press, W, Teukolsky, S, et al., "Numerical Recipes in C", 2nd Ed,
            pp 275-277, Cambridge Press, 1996.

    Parameters:
        min - the minimum number that can be generated
        max - the maximum number that can be generated

    Return Value:
        A random number between min and max inclusively.

*****************************************************************************/

inline double RNG::uniformRandCLib(double _min,double _max) {

    return _min + (_max-_min)*((double)rand())/((double)RAND_MAX);
}


/*! ************************************************************************* 

    Function:
        CRandomNumberGenerator::UniformRand()

    Description:
        Generates a random number between min and max exclusively using a
        uniform random deviate.  The actual range of number is [min+EPS..max-EPS],
        where EPS is defined above.

    Parameters:
        min - the minimum number that can be generated
        max - the maximum number that can be generated

    Return Value:
        A random number between min and max exclusively.

    Notes:
        This algoirthm is adapted from:

            Press, W, Teukolsky, S, et al., "Numerical Recipes in C", 2nd Ed,
            pp 278-280, Cambridge Press, 1996.

*****************************************************************************/

inline double RNG::uniformRand(double _min, double _max) {


    int j;
    int k;
    double temp;
    static int initial=1;
    static int iy=0;
    static int iv[NTAB];


    if( initial ) {                            // first time the function is called make sure the seed is a negative number
        if( seed > 0 ) {
            seed = -seed;
        }
        initial = 0;
    }

    if( (seed <= 0) || (!iy) ) {             // initalize

        if( -seed < 1) {                   // don't allow the seed to be 0 or else
            seed = 1;                        // the generator will always return 0
        }
        else {
            seed = -seed;
        }

        for(j=NTAB+7; j>=0; j--) {
            k = seed / IQ;
            seed = IA*(seed-k*IQ)-IR*k;

            if( seed < 0 ) {
                seed += IM;
            }

            if( j < NTAB ) {
                iv[j] = seed;
            }
        }

        iy = iv[0];
    }

    k = seed/IQ;                          // start here when not initalizing
    seed = IA*(seed-k*IQ)-IR*k;         // compute idum = (IA*idnum) % IM without overflows by Schrage's method
    if( seed < 0 ) {
        seed += IM;
    }

    j = iy/(1 + (IM-1)/NTAB);                            // in the range 0..NTAB-1
    iy = iv[j];                             // output previously stored value and refill the shuffle table
    iv[j] = seed;

    if( (temp=(1.0f/IM)*iy) > RNMX ) {             // user's don't expect the endpoint values
        return (double) (_min + (_max-_min)*RNMX);
    }
    else {
        return (double) (_min + (_max-_min)*temp);
    }

}


/*! ************************************************************************* 

    Function:
        CRandomNumberGenerator::NormalRand()

    Description:
        Generates a random number between min and max inclusively using a
        Normal random deviate with mean mu and standard deviation sigma.  
        The algorithm is based on the transformation method of probability
        distributions.

    Parameters:
        None

    Return Value:
        A random number between min and max inclusively.

    Notes:
        This algoirthm is adapted from:

            Press, W, Teukolsky, S, et al., "Numerical Recipes in C", 2nd Ed,
            pp 287-290, Cambridge Press, 1996.

*****************************************************************************/


inline double RNG::normalRand(void) {

    static int iset=0;
    static double gset;
    double fac, rsq, v1, v2;


    if( iset == 0 ) {
        do {
            v1 = 2.0f * uniformRand(0.0f, 1.0f) - 1.0f;      // pick two uniform numbrs in the square
            v2 = 2.0f * uniformRand(0.0f, 1.0f) - 1.0f;      // extending from -1 to +1 in each direction
            
            rsq = v1*v1 + v2*v2;                             
        } while( (rsq >= 1.0) || (rsq == 0.0) );             // see if they are in the unit circle

        fac = sqrtf(-2.0f * (double) log( (double) rsq) / rsq );
        
        // make the Box-Muller transformation to get two normal deviates.  
        // Return one and save the other for next time.

        gset = v1*fac;
        iset = 1;                                            // set flag because we have two numbers

        return v2*fac;
    }
    else {
        iset = 0;                                            // unset flag because we are using the other number
        return gset;        
    }

}


/*****************************************************************************/


inline double RNG::normalRand(double mu, double sigma) {

    return (normalRand()*sigma) + mu;

}
#endif
