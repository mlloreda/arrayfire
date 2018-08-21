/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/graphics.h>
#include <af/data.h>

#include <common/ArrayInfo.hpp>
#include <common/graphics_common.hpp>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <handle.hpp>
#include <join.hpp>
#include <reduce.hpp>
#include <transpose.hpp>
#include <vector_field.hpp>

#include <vector>

using std::vector;
using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;

template<typename T>
forge::Chart* setup_vector_field(const forge::Window* const window,
                                 const vector<af_array>& points, const vector<af_array>& directions,
                                 const af_cell* const props, const bool transpose_ = true)
{
    vector< Array<T> > pnts;
    vector< Array<T> > dirs;

    for (unsigned i=0; i<points.size(); ++i) {
        pnts.push_back(getArray<T>(points[i]));
        dirs.push_back(getArray<T>(directions[i]));
    }

    // Join for set up vector
    Array<T> pIn = detail::join(1, pnts);
    Array<T> dIn = detail::join(1, dirs);

    // do transpose if required
    if(transpose_) {
        pIn = transpose<T>(pIn, false);
        dIn = transpose<T>(dIn, false);
    }

    ForgeManager& fgMngr = ForgeManager::getInstance();

    // Get the chart for the current grid position (if any)
    forge::Chart* chart = NULL;

    if(pIn.dims()[0] == 2) {
        if (props->col>-1 && props->row>-1)
            chart = fgMngr.getChart(window, props->row, props->col, FG_CHART_2D);
        else
            chart = fgMngr.getChart(window, 0, 0, FG_CHART_2D);
    } else {
        if (props->col>-1 && props->row>-1)
            chart = fgMngr.getChart(window, props->row, props->col, FG_CHART_3D);
        else
            chart = fgMngr.getChart(window, 0, 0, FG_CHART_3D);
    }

    forge::VectorField* vectorfield = fgMngr.getVectorField(chart, pIn.dims()[1], getGLType<T>());

    // ArrayFire LOGO dark blue shade
    vectorfield->setColor(0.130f, 0.173f, 0.263f, 1.0);

    // If chart axes limits do not have a manual override
    // then compute and set axes limits
    if(!fgMngr.getChartAxesOverride(chart)) {
        float cmin[3], cmax[3];
        T     dmin[3], dmax[3];
        chart->getAxesLimits(&cmin[0], &cmax[0], &cmin[1], &cmax[1], &cmin[2], &cmax[2]);
        copyData(dmin, reduce<af_min_t, T, T>(pIn, 1));
        copyData(dmax, reduce<af_max_t, T, T>(pIn, 1));

        if(cmin[0] == 0 && cmax[0] == 0
        && cmin[1] == 0 && cmax[1] == 0
        && cmin[2] == 0 && cmax[2] == 0) {
            // No previous limits. Set without checking
            cmin[0] = step_round(dmin[0], false);
            cmax[0] = step_round(dmax[0], true);
            cmin[1] = step_round(dmin[1], false);
            cmax[1] = step_round(dmax[1], true);
            if(pIn.dims()[0] == 3) cmin[2] = step_round(dmin[2], false);
            if(pIn.dims()[0] == 3) cmax[2] = step_round(dmax[2], true);
        } else {
            if(cmin[0] > dmin[0])       cmin[0] = step_round(dmin[0], false);
            if(cmax[0] < dmax[0])       cmax[0] = step_round(dmax[0], true);
            if(cmin[1] > dmin[1])       cmin[1] = step_round(dmin[1], false);
            if(cmax[1] < dmax[1])       cmax[1] = step_round(dmax[1], true);
            if(pIn.dims()[0] == 3) {
                if(cmin[2] > dmin[2])   cmin[2] = step_round(dmin[2], false);
                if(cmax[2] < dmax[2])   cmax[2] = step_round(dmax[2], true);
            }
        }

        if(pIn.dims()[0] == 2) {
            chart->setAxesLimits(cmin[0], cmax[0], cmin[1], cmax[1]);
        } else if(pIn.dims()[0] == 3) {
            chart->setAxesLimits(cmin[0], cmax[0], cmin[1], cmax[1], cmin[2], cmax[2]);
        }
    }

    copy_vector_field<T>(pIn, dIn, vectorfield);

    return chart;
}

af_err vectorFieldWrapper(const af_window wind, const af_array points, const af_array directions,
                          const af_cell* const props)
{
    if (wind == 0) {
        AF_RETURN_ERROR("Not a valid window", AF_SUCCESS);
    }

    try {
        ARG_SETUP(directions);
        ARG_SETUP(points);
        ASSERT_TYPE_EQ(points, directions);
        ASSERT_NDIM_EQ(points, 2);
        ASSERT_DIM(points, directions);
        ASSERT_DIM_EQ(points, 1, 2);

        const dim4 points_dims  = points_info.dims();
        DIM_ASSERT(0, points_dims[1] == 2 || points_dims[1] == 3); // Columns:P 2 means 2D and 3 means 3D // \TODO(miguel) what to do about this or???

        forge::Window* window = reinterpret_cast<forge::Window*>(wind);
        makeContextCurrent(window);

        forge::Chart* chart = NULL;
        vector<af_array> pnts{points};
        vector<af_array> dirs{directions};
        switch(points_info.getType()) {
            case f32: chart = setup_vector_field<float  >(window, pnts, dirs, props); break;
            case s32: chart = setup_vector_field<int    >(window, pnts, dirs, props); break;
            case u32: chart = setup_vector_field<uint   >(window, pnts, dirs, props); break;
            case s16: chart = setup_vector_field<short  >(window, pnts, dirs, props); break;
            case u16: chart = setup_vector_field<ushort >(window, pnts, dirs, props); break;
            case u8 : chart = setup_vector_field<uchar  >(window, pnts, dirs, props); break;
            default:  TYPE_ERROR(points);
        }

        auto gridDims = ForgeManager::getInstance().getWindowGrid(window);
        // Window's draw function requires either image or chart
        if (props->col > -1 && props->row > -1)
            window->draw(gridDims.first, gridDims.second,
                         props->row * gridDims.second + props->col,
                         *chart, props->title);
        else
            window->draw(*chart);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err vectorFieldWrapper(const af_window wind,
                          const af_array xPoints, const af_array yPoints, const af_array zPoints,
                          const af_array xDirs, const af_array yDirs, const af_array zDirs,
                          const af_cell* const props)
{
    if (wind == 0) {
        AF_RETURN_ERROR("Not a valid window", AF_SUCCESS);
    }

    try {
        ARG_SETUP(xPoints); ARG_SETUP(yPoints); ARG_SETUP(zPoints);
        ARG_SETUP(xDirs); ARG_SETUP(yDirs); ARG_SETUP(zDirs);

        // Assert all arrays are equal dimensions
        ASSERT_DIM(xPoints, xDirs);
        ASSERT_DIM(yPoints, yDirs);
        ASSERT_DIM(zPoints, zDirs);
        ASSERT_DIM(xPoints, yPoints);
        ASSERT_DIM(xPoints, zPoints);

        // Verify vector
        ASSERT_NDIM_EQ(xPoints, 1);

        // Assert all arrays are equal types
        ASSERT_TYPE_EQ(xPoints, xDirs);
        ASSERT_TYPE_EQ(yPoints, yDirs);
        ASSERT_TYPE_EQ(zPoints, zDirs);
        ASSERT_TYPE_EQ(xPoints, yPoints);
        ASSERT_TYPE_EQ(xPoints, zPoints);

        forge::Window* window = reinterpret_cast<forge::Window*>(wind);
        makeContextCurrent(window);

        forge::Chart* chart = NULL;

        vector<af_array> points{xPoints, yPoints, zPoints};
        vector<af_array> directions{xDirs, yDirs, zDirs};

        switch(xPoints_info.getType()) {
            case f32: chart = setup_vector_field<float  >(window, points, directions, props); break;
            case s32: chart = setup_vector_field<int    >(window, points, directions, props); break;
            case u32: chart = setup_vector_field<uint   >(window, points, directions, props); break;
            case s16: chart = setup_vector_field<short  >(window, points, directions, props); break;
            case u16: chart = setup_vector_field<ushort >(window, points, directions, props); break;
            case u8 : chart = setup_vector_field<uchar  >(window, points, directions, props); break;
            default:  TYPE_ERROR(xPoints);
        }

        auto gridDims = ForgeManager::getInstance().getWindowGrid(window);
        // Window's draw function requires either image or chart
        if (props->col > -1 && props->row > -1)
            window->draw(gridDims.first, gridDims.second,
                         props->row * gridDims.second + props->col,
                         *chart, props->title);
        else
            window->draw(*chart);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err vectorFieldWrapper(const af_window wind,
                          const af_array xPoints, const af_array yPoints,
                          const af_array xDirs, const af_array yDirs,
                          const af_cell* const props)
{
    if (wind == 0) {
        AF_RETURN_ERROR("Not a valid window", AF_SUCCESS);
    }

    try {
        ARG_SETUP(xPoints); ARG_SETUP(yPoints);
        ARG_SETUP(xDirs); ARG_SETUP(yDirs);
        ASSERT_DIM(xPoints, xDirs);
        ASSERT_DIM(yPoints, yDirs);
        ASSERT_DIM(xPoints, yPoints);
        ASSERT_NDIM_EQ(xPoints, 1);
        ASSERT_TYPE_EQ(xPoints, xDirs);
        ASSERT_TYPE_EQ(yPoints, yDirs);
        ASSERT_TYPE_EQ(xPoints, yPoints);

        forge::Window* window = reinterpret_cast<forge::Window*>(wind);
        makeContextCurrent(window);

        forge::Chart* chart = NULL;
        vector<af_array> points{xPoints, yPoints};
        vector<af_array> directions{xDirs, yDirs};
        switch(xPoints_info.getType()) {
            case f32: chart = setup_vector_field<float  >(window, points, directions, props); break;
            case s32: chart = setup_vector_field<int    >(window, points, directions, props); break;
            case u32: chart = setup_vector_field<uint   >(window, points, directions, props); break;
            case s16: chart = setup_vector_field<short  >(window, points, directions, props); break;
            case u16: chart = setup_vector_field<ushort >(window, points, directions, props); break;
            case u8 : chart = setup_vector_field<uchar  >(window, points, directions, props); break;
            default:  TYPE_ERROR(xPoints);
        }

        auto gridDims = ForgeManager::getInstance().getWindowGrid(window);
        // Window's draw function requires either image or chart
        if (props->col > -1 && props->row > -1)
            window->draw(gridDims.first, gridDims.second,
                         props->row * gridDims.second + props->col,
                         *chart, props->title);
        else
            window->draw(*chart);
    }
    CATCHALL;
    return AF_SUCCESS;
}

#endif // WITH_GRAPHICS

// ADD THIS TO UNIFIED
af_err af_draw_vector_field_nd(const af_window wind,
                const af_array points, const af_array directions,
                const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    return vectorFieldWrapper(wind, points, directions, props);
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_draw_vector_field_3d(
                const af_window wind,
                const af_array xPoints, const af_array yPoints, const af_array zPoints,
                const af_array xDirs, const af_array yDirs, const af_array zDirs,
                const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    return vectorFieldWrapper(wind, xPoints, yPoints, zPoints, xDirs, yDirs, zDirs, props);
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_draw_vector_field_2d(
                const af_window wind,
                const af_array xPoints, const af_array yPoints,
                const af_array xDirs, const af_array yDirs,
                const af_cell* const props)
{
#if defined(WITH_GRAPHICS)
    return vectorFieldWrapper(wind, xPoints, yPoints, xDirs, yDirs, props);
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}
