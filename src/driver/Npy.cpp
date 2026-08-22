#include "Npy.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tensora::npy {

namespace {

// The 6-byte magic string every .npy file starts with.
const char kMagic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

// Extracts the substring after `key` up to the next `,` or `}`. Used to
// read one value out of the header's Python-dict text. Returns "" if
// the key is absent.
std::string fieldAfter(const std::string& header, const std::string& key) {
  size_t k = header.find(key);
  if (k == std::string::npos) return "";
  size_t start = k + key.size();
  size_t end = header.find_first_of(",}", start);
  if (end == std::string::npos) end = header.size();
  return header.substr(start, end - start);
}

// Removes spaces, single quotes, and parentheses from `s`.
std::string strip(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c != ' ' && c != '\'' && c != '(' && c != ')') out += c;
  }
  return out;
}

}  // namespace

bool read(const std::string& path, NpyArray& out, std::string& err) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    err = "cannot open '" + path + "'";
    return false;
  }

  char magic[6];
  in.read(magic, 6);
  if (in.gcount() != 6 || std::memcmp(magic, kMagic, 6) != 0) {
    err = "'" + path + "' is not a .npy file (bad magic)";
    return false;
  }

  unsigned char version[2];
  in.read(reinterpret_cast<char*>(version), 2);
  if (in.gcount() != 2 || version[0] != 1) {
    err = "'" + path + "' is not a version 1.0 .npy file";
    return false;
  }

  unsigned char lenBytes[2];
  in.read(reinterpret_cast<char*>(lenBytes), 2);
  if (in.gcount() != 2) {
    err = "'" + path + "' has a truncated header";
    return false;
  }
  // HEADER_LEN is a little-endian uint16.
  size_t headerLen = static_cast<size_t>(lenBytes[0]) | (static_cast<size_t>(lenBytes[1]) << 8);

  std::string header(headerLen, '\0');
  in.read(header.data(), static_cast<std::streamsize>(headerLen));
  if (static_cast<size_t>(in.gcount()) != headerLen) {
    err = "'" + path + "' has a truncated header";
    return false;
  }

  // Only little-endian float32 is supported. `<f4` is what numpy writes
  // on a little-endian machine; `=f4` is the native-order spelling.
  std::string descr = strip(fieldAfter(header, "'descr':"));
  if (descr != "<f4" && descr != "=f4") {
    err = "'" + path + "' has dtype '" + descr + "'; only float32 ('<f4') is supported";
    return false;
  }

  std::string fortran = strip(fieldAfter(header, "'fortran_order':"));
  if (fortran != "False") {
    err = "'" + path + "' is Fortran-ordered; only C-contiguous arrays are supported";
    return false;
  }

  // Parse the shape tuple, for example `(4, 8)` -> {4, 8}. The tuple
  // itself contains commas, so read the text between its parentheses
  // rather than up to the next comma. A 0-d array has an empty tuple
  // and one element.
  std::string shapeText;
  {
    size_t k = header.find("'shape':");
    size_t lp = (k == std::string::npos) ? std::string::npos : header.find('(', k);
    size_t rp = (lp == std::string::npos) ? std::string::npos : header.find(')', lp);
    if (lp == std::string::npos || rp == std::string::npos) {
      err = "'" + path + "' has an unreadable shape field";
      return false;
    }
    shapeText = strip(header.substr(lp + 1, rp - lp - 1));
  }
  out.shape.clear();
  int64_t count = 1;
  std::stringstream ss(shapeText);
  std::string dim;
  while (std::getline(ss, dim, ',')) {
    if (dim.empty()) continue;
    int64_t d = std::stoll(dim);
    out.shape.push_back(d);
    count *= d;
  }

  out.data.resize(static_cast<size_t>(count));
  in.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(count) * 4);
  if (in.gcount() != static_cast<std::streamsize>(count) * 4) {
    err = "'" + path + "' has fewer data bytes than its shape declares";
    return false;
  }
  return true;
}

bool write(const std::string& path, const float* data, const std::vector<int64_t>& shape, std::string& err) {
  std::ofstream os(path, std::ios::binary);
  if (!os) {
    err = "cannot write '" + path + "'";
    return false;
  }

  // Build the shape tuple. numpy writes a trailing comma for a 1-tuple,
  // e.g. `(4,)`; a plain `(4)` also parses, so a uniform `n,` per dim
  // is safe.
  std::string shapeText = "(";
  int64_t count = 1;
  for (int64_t d : shape) {
    shapeText += std::to_string(d) + ",";
    count *= d;
  }
  shapeText += ")";

  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shapeText + ", }";
  // The total header (10 fixed bytes + dict + newline) must be a
  // multiple of 64; pad the dict with spaces before the newline.
  size_t unpadded = 10 + dict.size() + 1;
  size_t pad = (64 - (unpadded % 64)) % 64;
  dict += std::string(pad, ' ');
  dict += '\n';

  os.write(kMagic, 6);
  unsigned char version[2] = {1, 0};
  os.write(reinterpret_cast<char*>(version), 2);
  size_t headerLen = dict.size();
  unsigned char lenBytes[2] = {static_cast<unsigned char>(headerLen & 0xFF),
                               static_cast<unsigned char>((headerLen >> 8) & 0xFF)};
  os.write(reinterpret_cast<char*>(lenBytes), 2);
  os.write(dict.data(), static_cast<std::streamsize>(dict.size()));
  os.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count) * 4);
  if (!os) {
    err = "failed while writing '" + path + "'";
    return false;
  }
  return true;
}

}  // namespace tensora::npy
