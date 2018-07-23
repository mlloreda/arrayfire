/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace af;

int main(int argc, char *argv[])
{
    try {


        // Select a device and display arrayfire info
        int device = argc > 1 ? atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        int m = 3;
        int n = 2;
        array A = af::constant(0, dim4(m,n));
        af_print(A);

        array spA = af::sparse(A);
        af_print(spA);

    } catch (af::exception& e) {

        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
