#include "simulation/cpu/GridShiftHelper.h"
#include <cmath>

GridShiftHelper::GridShiftHelper(const GridTopology& topology) {
    initStorageDimensions(topology);
    buildValidityMask(topology);
    precomputeNeighborShifts(topology);
}

void GridShiftHelper::initStorageDimensions(const GridTopology& topology) {
    const StorageCoord storageDims = topology.storageDim;
    storageWidth = storageDims.x;
    storageHeight = storageDims.y;
}

void GridShiftHelper::buildValidityMask(const GridTopology& topology) {
    validityMask.resize(storageHeight, storageWidth);
    validityMask.setZero();
    
    for (int y = 0; y < storageHeight; ++y) {
        for (int x = 0; x < storageWidth; ++x) {
            validityMask(y, x) = topology.isValid(StorageCoord{x, y}) ? 1.0f : 0.0f;
        }
    }
}

void GridShiftHelper::precomputeNeighborShifts(const GridTopology& topology) {
    for (size_t i = 0; i < NUM_DIRECTIONS; ++i) {
        const auto& off = topology.neighborOffsets[i];
        outgoingShifts[i] = computeShiftInfo(-off.r, -off.q);
        incomingShifts[i] = computeShiftInfo(+off.r, +off.q);
    }
}

ShiftInfo GridShiftHelper::computeShiftInfo(int dy, int dx) const {
    ShiftInfo s;
    s.dstRow = std::max(dy, 0);
    s.dstCol = std::max(dx, 0);
    s.srcRow = std::max(-dy, 0);
    s.srcCol = std::max(-dx, 0);
    s.copyH = std::max(storageHeight - std::abs(dy), 0);
    s.copyW = std::max(storageWidth - std::abs(dx), 0);
    return s;
}

void GridShiftHelper::shiftMatrix(const MatrixXf& src, MatrixXf& dst, const ShiftInfo& s) const {
    dst.setZero();
    if (s.isValid()) {
        dst.block(s.dstRow, s.dstCol, s.copyH, s.copyW) = 
            src.block(s.srcRow, s.srcCol, s.copyH, s.copyW);
    }
}
