/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <utility.hpp>
#include <tuple>
#include <unordered_map>
#include <cstring>

// \TODO use templates
namespace cpu
{
namespace kernel
{

using Coordinate = std::tuple<unsigned,unsigned>;

#define print(var) std::cout << "[INFO] " << #var << ": "  << var << std::endl;


std::unordered_map<std::string, int> searchedCoordMap;
std::unordered_map<std::string, int> searchCoordMap; // \TODO
std::unordered_map<std::string, int> regionCoordMap;

// Utility functions
template<typename T>
void PrintNeighborhoodValues(std::vector<T> &n)
{
    for (int i = 0; i < n.size(); ++i) {
        std::cout << "  " << n[i] << " ";
    }
}

template<typename T>
void PrintRegion(const std::vector<Coordinate> &n)
{
    int row, col;
    for (typename std::vector<T>::iterator it = n.begin(); it != n.end(); ++it) {
        std::tie(row, col) = *it;
        std::cout << "(" << row << ", " << col << ")" << std::endl;
    }
}

void printCoordinate(std::string msg, int row, int col)
{
    std::cout << msg << "(" << row << ", " << col << ")\n";
}

std::string CoordinateToString(const Coordinate &c)
{
    int first, second;
    std::tie(first, second) = c;
    return std::to_string(first) + std::to_string(second);
}

template<typename T>
std::tuple<float,float> calculateThresholdsFromNeighborhoodStats(
        T intensity,
        std::tuple<float,float> stats,
        unsigned multiplier)
{
    float mean = 0, std_dev = 0;
    std::tie(mean, std_dev) = stats;
    short lowestIntensity = std::numeric_limits<short>::max();
    short highestIntensity = -lowestIntensity;
    if (lowestIntensity > intensity)
        lowestIntensity = intensity;
    if (highestIntensity < intensity)
        highestIntensity = intensity;

    float lower = mean - multiplier * std_dev;
    float upper = mean + multiplier * std_dev;
    if (lower > lowestIntensity)
        lower = lowestIntensity;
    else if (upper < highestIntensity)
        upper = highestIntensity;

    return std::make_pair(lower, upper);
}

void AddCoordNeighborsToSearchSpace(
        std::deque<Coordinate> &searchSpace,
        std::vector<Coordinate> &region,
        const Coordinate &coord,
        const af_connectivity conn,
        const unsigned radius, dim4 dims)
{
    unsigned coordRow, coordCol;
    std::tie(coordRow, coordCol) = coord;

    int neighborhoodDiameter = 2*radius+1;
    int rowOffset = coordRow - (neighborhoodDiameter/2);
    int colOffset = coordCol - (neighborhoodDiameter/2);
    int neighRow = 0, neighCol = 0;

    for (int col = 0; col < neighborhoodDiameter; ++col) {
        for (int row = 0; row < neighborhoodDiameter; ++row) {
            neighRow = rowOffset + row;
            neighCol = colOffset + col;

            if ((neighRow < 0) || (neighRow > dims[0])) continue;
            if ((neighCol < 0) || (neighCol > dims[1])) continue;
            if (conn == AF_CONNECTIVITY_4) {
                if ((neighRow != coordRow) && (neighCol != coordCol)) continue;
            }
            Coordinate neighCoord  = std::make_pair(neighRow, neighCol);
            std::string strNeighCoord = CoordinateToString(neighCoord);
            bool isCoordInSearch = std::find(searchSpace.begin(), searchSpace.end(), neighCoord) != searchSpace.end();
            bool isCoordInRegion = (regionCoordMap.find(strNeighCoord) != regionCoordMap.end());
            if(!isCoordInSearch && !isCoordInRegion) {
                searchSpace.push_back(neighCoord);
            }
        }
    }
}

template<typename T>
void ProcessCoordinate(const Coordinate &coord,
                       const std::tuple<float,float> &thresholds,
                       T *inPtr,
                       const af_connectivity conn,
                       std::vector<Coordinate> &region,
                       std::deque<Coordinate> &search,
                       std::vector<Coordinate> &searched,
                       const unsigned radius, const dim4 dims)
{
    // \TODO cache
    float lower, upper;
    std::tie(lower, upper) = thresholds;

    std::string strCoord = CoordinateToString(coord);
    bool is_coord_in_searched = searchedCoordMap[strCoord];
    if (!is_coord_in_searched) {
        unsigned coord_row, coord_col;
        std::tie(coord_row, coord_col) = coord;
        T intensity = inPtr[coord_col * dims[0] + coord_row];

        if (intensity >= lower && intensity <= upper &&
            coord_row < dims[0] && coord_col < dims[1]) {
            region.push_back(coord);
            regionCoordMap.insert({strCoord, 1});
            AddCoordNeighborsToSearchSpace(search, region, coord, conn, radius, dims);
        }
        // searched.push_back(strCoord);
        searchedCoordMap.insert({strCoord, 1});
    }
}

template<typename T>
void confidenceConnected(Param<T> out, CParam<T> in, CParam<T> seed,
                         const af::ccType cc_method,
                         const unsigned radius, const unsigned multiplier, int iter)
{
    dim4 oDims    = out.dims();
    assert(oDims[0] > 1 && oDims[1] > 1 && oDims[2] == 1 && oDims[3] == 1);

    const T *inPtr = in.get();

    // 0. Get the seed row and column
    const T *seedPtr = seed.get();
    const unsigned seedRow = (unsigned)seedPtr[0];
    const unsigned seedCol = (unsigned)seedPtr[1];

    // NOTE: Want this all the way up here so we can populate the
    // search space as we're calculating statistics for the intial
    // round.
    std::deque<Coordinate> search;

    // 1. Calculate seed's neighborhood statistics on the neighborhood
    // intensity values gathered in previous step
    // \TODO Assuming this neighborhood is 8-way connected. Is this a sane
    // assumption?
    std::tuple<float, float> stats = [&inPtr, &oDims, seedRow, seedCol, radius](std::deque<Coordinate> &search) -> std::tuple<float,float> {
        int neighborhoodDiameter = 2*radius+1;
        int imageRowOffset = seedRow - (neighborhoodDiameter/2);
        int imageColOffset = seedCol - (neighborhoodDiameter/2);
        int imageRow, imageCol = 0;

        float count = 0, acc = 0, sumOfSquares = 0, mean = 0, variance = 0;
        for (int col = 0; col < neighborhoodDiameter; ++col) {
            for (int row = 0; row < neighborhoodDiameter; ++row) {
                imageRow = imageRowOffset + row;
                imageCol = imageColOffset + col;

                search.push_back(std::make_tuple(imageRow, imageCol));

                T intensity = inPtr[imageCol * oDims[0] + imageRow];
                acc += intensity;
                sumOfSquares += (intensity*intensity);
                count++;
            }
        }
        assert(count == (neighborhoodDiameter)*(neighborhoodDiameter));
        assert(search.size() == (2*radius+1)*(2*radius+1));

        mean = float(acc/count);
        variance = (sumOfSquares - (acc*acc / count)) / (count - 1.0);
        float stddev = sqrt(variance);
        // DEBUGGING
        // print(sumOfSquares);
        // print(mean);
        // print(variance);
        // print(stddev);
        return std::make_tuple(mean, sqrt(variance));
    }(search);

    float mean, std_dev;
    std::tie(mean, std_dev) = stats;

    // 2. Calculate and adjust thresholds
    short seedIntensity = (short)inPtr[seedCol*oDims[0] + seedRow];
    std::tuple<float,float> thresholds = calculateThresholdsFromNeighborhoodStats(seedIntensity, stats, multiplier);

    // 3. Return binary output from flood fill operation
    std::vector<Coordinate> region;
    std::vector<Coordinate> searched;

    auto start = std::chrono::system_clock::now();
    while(!search.empty()) {
        Coordinate coord = search.front();
        search.pop_front();
        ProcessCoordinate(coord, thresholds, inPtr, AF_CONNECTIVITY_8, region, search, searched, radius, oDims);
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::cout << "Time elapsed in `ProcessCoordinate()` loop: " << elapsed.count() << std::endl;


    std::cout << "----------------\n";
    // \TODO play around with the total number of iterations.
    const int numberOfIterations = (iter-1 >= 0) ? iter-1 : 0;
    std::cout << "numberOfIterations: " << numberOfIterations << std::endl;
    for (int iter = 0; iter < numberOfIterations; ++iter) {
        int neighborhoodSize = region.size();
        float acc = 0, sumOfSquares = 0, mean = 0, variance = 0, std_dev = 0;
        for (int neigh = 0; neigh < neighborhoodSize; ++neigh) {
            unsigned neighRow, neighCol;
            std::tie(neighRow, neighCol) = region[neigh];
            T neighIntensity = inPtr[neighCol*oDims[0] + neighRow];
            acc += neighIntensity;
            sumOfSquares += (neighIntensity * neighIntensity);
        }
        mean = acc / neighborhoodSize;
        variance = (sumOfSquares - (acc*acc / neighborhoodSize)) / (neighborhoodSize - 1.0);
        std_dev = sqrt(variance);

        // DEBUGGING
        // print(sumOfSquares);
        // print(mean);
        // print(variance);
        // print(std_dev);

        std::tuple<float,float> neighStats = std::make_pair(mean,std_dev);
        std::tuple<float,float> thresholds = calculateThresholdsFromNeighborhoodStats(seedIntensity, neighStats, multiplier);

        std::deque<Coordinate> searchSpace;
        std::vector<Coordinate> searched;
        std::vector<Coordinate> new_region; // Need to store our gnew region elsewhere

        // Populate search space.
        // -
        // Initialize our search space with connected coordinates
        // found in first pass whose intensity values fall within
        // range.
        searchSpace.push_back(std::make_pair(seedRow, seedCol));
        // \NOTE We need this loop because we can no longer assume an
        // 8-way connected neighborhood around the seed point.
        for (auto currNode : region) {
            unsigned currRow, currCol;
            std::tie(currRow, currCol) = currNode;
            bool isWest  = ((currRow == seedRow  ) && (currCol == seedCol-1));
            bool isNorth = ((currRow == seedRow-1) && (currCol == seedCol  ));
            bool isSouth = ((currRow == seedRow+1) && (currCol == seedCol  ));
            bool isEast  = ((currRow == seedRow  ) && (currCol == seedCol+1));
            if (isWest || isNorth || isSouth || isEast)
                searchSpace.push_back(currNode);
        }
        while (!searchSpace.empty()) {
            Coordinate currCoord = searchSpace.front();
            searchSpace.pop_front();
            ProcessCoordinate(currCoord, thresholds, inPtr, AF_CONNECTIVITY_4, new_region, searchSpace, searched, radius, oDims);
        }

        // Delete items in `region` vector and...
        region.clear();
        // regionCoordMap.clear();
        // ...swap with items from the `new_region` vector,
        std::swap(region, new_region);

        bool last_loop = (iter == numberOfIterations-1);
        if (last_loop) {
            T* dst = out.get();
            std::memset(dst, 0, oDims[0]*oDims[1]);
            for (int idx = 0; idx < region.size(); ++idx) {
                unsigned outRow, outCol;
                std::tie(outRow, outCol) = region[idx];
                dst[outCol * oDims[0] + outRow] = (T)255;
            }
        }
    } // End iterations loop


    assert(numberOfIterations >= 0);
    if (numberOfIterations == 0) {
        T* dst = out.get();
        std::memset(dst, 0, oDims[0]*oDims[1]);
        for (int idx = 0; idx < region.size(); ++idx) {
            unsigned outRow, outCol;
            std::tie(outRow, outCol) = region[idx];
            dst[outCol * oDims[0] + outRow] = (T)255;
        }
    }
}

}
}
