# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/raymondtay/GPUMLib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/raymondtay/GPUMLib

# Include any dependencies generated for this target.
include src/examples/RBF/CMakeFiles/RBF.dir/depend.make

# Include the progress variables for this target.
include src/examples/RBF/CMakeFiles/RBF.dir/progress.make

# Include the compile flags for this target's objects.
include src/examples/RBF/CMakeFiles/RBF.dir/flags.make

src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o: src/examples/RBF/CMakeFiles/RBF.dir/flags.make
src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o: src/examples/RBF/RBF.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/raymondtay/GPUMLib/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o"
	cd /Users/raymondtay/GPUMLib/src/examples/RBF && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/RBF.dir/RBF.cpp.o -c /Users/raymondtay/GPUMLib/src/examples/RBF/RBF.cpp

src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RBF.dir/RBF.cpp.i"
	cd /Users/raymondtay/GPUMLib/src/examples/RBF && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/raymondtay/GPUMLib/src/examples/RBF/RBF.cpp > CMakeFiles/RBF.dir/RBF.cpp.i

src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RBF.dir/RBF.cpp.s"
	cd /Users/raymondtay/GPUMLib/src/examples/RBF && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/raymondtay/GPUMLib/src/examples/RBF/RBF.cpp -o CMakeFiles/RBF.dir/RBF.cpp.s

src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o.requires:
.PHONY : src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o.requires

src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o.provides: src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o.requires
	$(MAKE) -f src/examples/RBF/CMakeFiles/RBF.dir/build.make src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o.provides.build
.PHONY : src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o.provides

src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o.provides.build: src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o

src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o: src/examples/RBF/CMakeFiles/RBF.dir/flags.make
src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o: src/examples/Dataset/Dataset.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/raymondtay/GPUMLib/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o"
	cd /Users/raymondtay/GPUMLib/src/examples/RBF && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o -c /Users/raymondtay/GPUMLib/src/examples/Dataset/Dataset.cpp

src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.i"
	cd /Users/raymondtay/GPUMLib/src/examples/RBF && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/raymondtay/GPUMLib/src/examples/Dataset/Dataset.cpp > CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.i

src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.s"
	cd /Users/raymondtay/GPUMLib/src/examples/RBF && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/raymondtay/GPUMLib/src/examples/Dataset/Dataset.cpp -o CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.s

src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o.requires:
.PHONY : src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o.requires

src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o.provides: src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o.requires
	$(MAKE) -f src/examples/RBF/CMakeFiles/RBF.dir/build.make src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o.provides.build
.PHONY : src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o.provides

src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o.provides.build: src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o

# Object files for target RBF
RBF_OBJECTS = \
"CMakeFiles/RBF.dir/RBF.cpp.o" \
"CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o"

# External object files for target RBF
RBF_EXTERNAL_OBJECTS =

src/examples/RBF/RBF: src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o
src/examples/RBF/RBF: src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o
src/examples/RBF/RBF: src/examples/RBF/CMakeFiles/RBF.dir/build.make
src/examples/RBF/RBF: /Developer/NVIDIA/CUDA-7.0/lib/libcudart.dylib
src/examples/RBF/RBF: src/libGPUMLibRBF.a
src/examples/RBF/RBF: /Developer/NVIDIA/CUDA-7.0/lib/libcublas.dylib
src/examples/RBF/RBF: /Developer/NVIDIA/CUDA-7.0/lib/libcudart.dylib
src/examples/RBF/RBF: src/examples/RBF/CMakeFiles/RBF.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable RBF"
	cd /Users/raymondtay/GPUMLib/src/examples/RBF && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RBF.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/examples/RBF/CMakeFiles/RBF.dir/build: src/examples/RBF/RBF
.PHONY : src/examples/RBF/CMakeFiles/RBF.dir/build

src/examples/RBF/CMakeFiles/RBF.dir/requires: src/examples/RBF/CMakeFiles/RBF.dir/RBF.cpp.o.requires
src/examples/RBF/CMakeFiles/RBF.dir/requires: src/examples/RBF/CMakeFiles/RBF.dir/__/Dataset/Dataset.cpp.o.requires
.PHONY : src/examples/RBF/CMakeFiles/RBF.dir/requires

src/examples/RBF/CMakeFiles/RBF.dir/clean:
	cd /Users/raymondtay/GPUMLib/src/examples/RBF && $(CMAKE_COMMAND) -P CMakeFiles/RBF.dir/cmake_clean.cmake
.PHONY : src/examples/RBF/CMakeFiles/RBF.dir/clean

src/examples/RBF/CMakeFiles/RBF.dir/depend:
	cd /Users/raymondtay/GPUMLib && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/raymondtay/GPUMLib /Users/raymondtay/GPUMLib/src/examples/RBF /Users/raymondtay/GPUMLib /Users/raymondtay/GPUMLib/src/examples/RBF /Users/raymondtay/GPUMLib/src/examples/RBF/CMakeFiles/RBF.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/examples/RBF/CMakeFiles/RBF.dir/depend

