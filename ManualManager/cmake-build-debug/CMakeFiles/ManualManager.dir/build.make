# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/ManualManager.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/ManualManager.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ManualManager.dir/flags.make

CMakeFiles/ManualManager.dir/main.cpp.o: CMakeFiles/ManualManager.dir/flags.make
CMakeFiles/ManualManager.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ManualManager.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ManualManager.dir/main.cpp.o -c /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/main.cpp

CMakeFiles/ManualManager.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ManualManager.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/main.cpp > CMakeFiles/ManualManager.dir/main.cpp.i

CMakeFiles/ManualManager.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ManualManager.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/main.cpp -o CMakeFiles/ManualManager.dir/main.cpp.s

# Object files for target ManualManager
ManualManager_OBJECTS = \
"CMakeFiles/ManualManager.dir/main.cpp.o"

# External object files for target ManualManager
ManualManager_EXTERNAL_OBJECTS =

ManualManager: CMakeFiles/ManualManager.dir/main.cpp.o
ManualManager: CMakeFiles/ManualManager.dir/build.make
ManualManager: CMakeFiles/ManualManager.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ManualManager"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ManualManager.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ManualManager.dir/build: ManualManager
.PHONY : CMakeFiles/ManualManager.dir/build

CMakeFiles/ManualManager.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ManualManager.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ManualManager.dir/clean

CMakeFiles/ManualManager.dir/depend:
	cd /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/cmake-build-debug /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/cmake-build-debug /Users/nikhil/Documents/Code/AmericanOptionPricing/ManualManager/cmake-build-debug/CMakeFiles/ManualManager.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ManualManager.dir/depend

