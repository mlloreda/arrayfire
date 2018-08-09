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

using namespace af;

int main(int argc, char *argv[])
{
    try {


        // Select a device and display arrayfire info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        printf("Create a 5-by-3 matrix of random floats on the GPU\n");
        array A = randu(5,3, f32);

        af::connectivity conn = static_cast<connectivity>(0);
        array reg = regions(A, conn);
        af_print(reg);
    } catch (af::exception& e) {

        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
