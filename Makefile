# Convenience wrapper around CMake. CMake is the authoritative build system;
# these targets just save typing. Run on WSL2 + Ubuntu (see docs/development).
BUILD ?= build

.PHONY: build test example clean

build:
	cmake -S . -B $(BUILD)
	cmake --build $(BUILD)

test: build
	ctest --test-dir $(BUILD) --output-on-failure

example: build
	./$(BUILD)/tensora --emit-tokens examples/mlp.tns

clean:
	rm -rf $(BUILD)
