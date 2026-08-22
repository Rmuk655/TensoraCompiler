#pragma once
#include <cstdint>
#include <string>
#include <vector>

// A small reader and writer for the NumPy `.npy` array format (version
// 1.0). The `tensora` driver uses this to load real input/weight
// values for `--run --data-dir` and to write the result tensor for
// `--output`. It supports only little-endian float32 (`<f4`),
// C-contiguous arrays -- the one element type Tensora uses (RM 3.23).
// This is a driver-only helper; no compiler phase depends on it.
namespace tensora::npy {

// A loaded array: flat C-contiguous float data plus its shape.
struct NpyArray {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

// Reads a float32 `.npy` file. Returns true on success. On failure,
// returns false and puts a human-readable reason in `err` (for example,
// a wrong dtype, a Fortran-order array, or a missing file).
bool read(const std::string& path, NpyArray& out, std::string& err);

// Writes `data` (row-major, length = product of `shape`) as a float32
// `.npy` file. Returns true on success; on failure returns false and
// sets `err`.
bool write(const std::string& path, const float* data, const std::vector<int64_t>& shape, std::string& err);

}  // namespace tensora::npy
