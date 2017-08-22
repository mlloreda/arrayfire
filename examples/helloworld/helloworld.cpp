/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace af;

int main(int argc, char *argv[])
{
    try {
        // array A = af::loadImage("/Users/mlloreda/work/af/af-connected-components/data/conf_cc_512x512.png", false).as(u8);


        unsigned a[25] = {232,105,76,216,134,
                          39,141,76,175,207,
                          213,141,219,32,244,
                          213,172,0,82,244,
                          197,87,216,82,130};
        array A(5, 5, a, afHost);
        A = A.as(u8);
        A = af::transpose(A);
        af_print(A);

        // > Seed
        unsigned s[2] = { 2, 2 };
        af::array seed(1,2, s, afHost);
        seed = seed.as(u8);

        // > Type of CC
        af_cc_type ccType;
        ccType = AF_CC_CONFIDENCE;


        // > Radius and multiplier
        unsigned radius = 1;
        unsigned multiplier = 1;
        int iter = 1;

        // \TODO: `af::confidence()` should take in a vector of seeds,
        // which should return a vector of neighborhoods with their
        // associated properties (radius, neighbors, lower and upper
        // thresholds, etc)
        double elapsedTime = 0;
        af::timer start = af::timer::start();

        array out = af::confidence(A, ccType, seed, radius, multiplier, iter);
        out.eval();
        af::sync();
        elapsedTime = af::timer::stop(start);
        // af_print(out);
        std::cout << "[INFO] Elapsed time: " << elapsedTime << std::endl;
        // af::Window wnd("ConfCC");
        // while(!wnd.close()) {
        //     wnd.image(out.as(u8));
        // }

    } catch (af::exception& e) {

        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
