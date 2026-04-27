#pragma once

#include "simulation/GridTopology.h"
#include <Eigen/Dense>
#include <array>

/**
 * @brief Information for shifting a matrix block by a neighbor offset.
 */
struct ShiftInfo {
    int dstRow, dstCol;
    int srcRow, srcCol;
    int copyH, copyW;
    
    [[nodiscard]] bool isValid() const { return copyH > 0 && copyW > 0; }
};

/**
 * @brief Precomputed shift operations and validity mask for hexagonal grid.
 * 
 * This class encapsulates the common grid operations needed by both
 * resource transfer and reproduction systems. It precomputes:
 * - Storage dimensions
 * - Validity mask (which storage cells are real hex cells)
 * - Shift info for all 6 neighbor directions (both outgoing and incoming)
 * 
 * Terminology:
 * - Outgoing shift (-offset): Used to check/affect neighbors FROM a cell
 * - Incoming shift (+offset): Used to receive/aggregate FROM neighbors TO a cell
 */
class GridShiftHelper {
public:
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    static constexpr int NUM_DIRECTIONS = 6;

    explicit GridShiftHelper(const GridTopology& topology);

    // Accessors
    [[nodiscard]] int width() const { return storageWidth; }
    [[nodiscard]] int height() const { return storageHeight; }
    [[nodiscard]] int size() const { return storageWidth * storageHeight; }
    
    [[nodiscard]] const MatrixXf& getValidityMask() const { return validityMask; }
    
    [[nodiscard]] const ShiftInfo& getOutgoingShift(int direction) const { 
        return outgoingShifts[direction]; 
    }
    [[nodiscard]] const ShiftInfo& getIncomingShift(int direction) const { 
        return incomingShifts[direction]; 
    }
    
    [[nodiscard]] const std::array<ShiftInfo, NUM_DIRECTIONS>& getOutgoingShifts() const {
        return outgoingShifts;
    }
    [[nodiscard]] const std::array<ShiftInfo, NUM_DIRECTIONS>& getIncomingShifts() const {
        return incomingShifts;
    }

    /**
     * @brief Shift src matrix and store in dst (dst = shifted(src)).
     */
    void shiftMatrix(const MatrixXf& src, MatrixXf& dst, const ShiftInfo& s) const;

    /**
     * @brief Accumulate shifted src into dst (dst += shifted(src)).
     */
    template <typename DerivedDst, typename DerivedSrc>
    void accumulateShifted(Eigen::MatrixBase<DerivedDst>& dst,
                           const Eigen::MatrixBase<DerivedSrc>& src,
                           const ShiftInfo& s) const {
        if (s.isValid()) {
            dst.block(s.dstRow, s.dstCol, s.copyH, s.copyW).noalias() +=
                src.block(s.srcRow, s.srcCol, s.copyH, s.copyW);
        }
    }

    /**
     * @brief Create a new matrix with proper dimensions, initialized to zero.
     */
    [[nodiscard]] MatrixXf createBuffer() const {
        MatrixXf buffer(storageHeight, storageWidth);
        buffer.setZero();
        return buffer;
    }

private:
    int storageWidth = 0;
    int storageHeight = 0;
    
    MatrixXf validityMask;
    std::array<ShiftInfo, NUM_DIRECTIONS> outgoingShifts;
    std::array<ShiftInfo, NUM_DIRECTIONS> incomingShifts;

    void initStorageDimensions(const GridTopology& topology);
    void buildValidityMask(const GridTopology& topology);
    void precomputeNeighborShifts(const GridTopology& topology);
    [[nodiscard]] ShiftInfo computeShiftInfo(int dy, int dx) const;
};
