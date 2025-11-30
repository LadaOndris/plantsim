#pragma once

#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "simulation/GridTopology.h"
#include "simulation/State.h"

/**
 * @brief Utility for printing State in ASCII format.
 */
class MapPrinter {
public:
  /**
   * @brief Prints the raw storage array as a grid.
   * 
   * @details Prints all cells including padding/dummy cells.
   * No hex offset alignment is applied.
   */
  template <typename T>
  static std::string printStorage(const GridTopology& topology,
                                  const std::vector<T>& data,
                                  std::function<char(T)> mapper) {
    std::ostringstream oss;
    StorageCoord dim = topology.getStorageDimension();

    for (int y = 0; y < dim.y; y++) {
      for (int x = 0; x < dim.x; x++) {
        int idx = y * dim.x + x;
        char c = (idx < static_cast<int>(data.size())) ? mapper(data[idx]) : '?';
        oss << c << ' ';
      }
      oss << '\n';
    }

    return oss.str();
  }

  /**
   * @brief Prints the hex map with offset alignment and validity masking.
   * 
   * @details Only prints valid hex cells. Invalid/padding cells are shown as spaces.
   * Odd rows are indented to show the hex stagger pattern.
   * Iterates over storage dimensions and converts to axial to check validity.
   */
  template <typename T>
  static std::string printHexMap(const GridTopology& topology,
                                 const std::vector<T>& data,
                                 std::function<char(T)> mapper) {
    std::ostringstream oss;
    StorageCoord dim = topology.getStorageDimension();

    for (int y = 0; y < dim.y; y++) {
      // Indent odd rows for hexagonal offset
      if (y % 2 == 1) {
        oss << ' ';
      }
      for (int x = 0; x < dim.x; x++) {
        AxialCoord axial = topology.storageToAxialCoord({x, y});
        
        if (topology.isValid(axial)) {
          int idx = y * dim.x + x;
          char c = (idx < static_cast<int>(data.size())) ? mapper(data[idx]) : '?';
          oss << c << ' ';
        }
      }
      oss << '\n';
    }

    return oss.str();
  }

  /**
   * @brief Prints State resources as hex map.
   */
  static std::string printHexMapResources(const GridTopology& topology, const State& state) {
    return printHexMap<float>(topology, state.resources,
                            [](float v) -> char {
                              if (v == 0)
                                return '.';
                              if (v <= 9)
                                return static_cast<char>('0' + v);
                              return '+';
                            });
  }

  /**
   * @brief Prints State cell types as hex map.
   */
  static std::string printHexMapCellTypes(const GridTopology& topology, const State& state) {
    return printHexMap<int>(topology, state.cellTypes,
                            [](int type) -> char {
                              switch (type) {
                              case 0:
                                return '.'; // Air
                              case 1:
                                return '#'; // Cell
                              default:
                                return '?';
                              }
                            });
  }

  /**
   * @brief Prints State resources as raw storage.
   */
  static std::string printStorageResources(const GridTopology& topology, const State& state) {
    return printStorage<float>(topology, state.resources,
                             [](int v) -> char {
                               if (v == 0)
                                 return '.';
                               if (v <= 9)
                                 return static_cast<char>('0' + v);
                               return '+';
                             });
  }

  /**
   * @brief Prints State cell types as raw storage.
   */
  static std::string printStorageCellTypes(const GridTopology& topology, const State& state) {
    return printStorage<int>(topology, state.cellTypes,
                             [](int type) -> char {
                               switch (type) {
                               case 0:
                                 return '.'; // Air
                               case 1:
                                 return '#'; // Cell
                               default:
                                 return '?';
                               }
                             });
  }
};
