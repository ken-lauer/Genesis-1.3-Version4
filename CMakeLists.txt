cmake_minimum_required(VERSION 3.12)
project(Genesis_1_3_Version4 CXX C)

option(USE_DPI "Use diagnostic plugin interface" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # replayed by the vec-audit target
set(CMAKE_CXX_EXTENTIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
# set(CMAKE_INSTALL_RPATH "/usr/local/lib")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# developer options
set(CMAKE_VERBOSE_MAKEFILE ON)
add_compile_options(-g)

# OpenMPI: preprocessor macro to disable C++ interface (then we don't have to link libmpi_cxx)
add_compile_definitions(OMPI_SKIP_MPICXX)

list(APPEND CMAKE_PREFIX_PATH $ENV{FFTW_ROOT})

# container for needed librarires
list(APPEND _libraries)

# Check for macOS with MacPorts
if(EXISTS "/opt/local/bin/h5pcc")
  set(CMAKE_CXX_COMPILER /opt/local/bin/h5pcc)
endif()


if (CMAKE_CXX_COMPILER MATCHES  "h5pcc")
   Message(STATUS "h5pcc compiler wrapper found - No additional libraries needed")
else()
   Message(STATUS "h5pcc not found - looking for basic compiler")
   Message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
   Message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")

   # check for MPI
   find_package(MPI REQUIRED)
   include_directories(${MPI_INCLUDE_PATH})
   Message(STATUS "MPI Found")

   # and check for HDF5
   SET(HDF5_PREFER_PARALLEL TRUE)
   find_package(HDF5 REQUIRED)
   Message(STATUS "HDF5 Found")
   Message(STATUS "Version: ${HDF5_VERSION}")
   if (TARGET HDF5::HDF5)
#       list(APPEND _libraries MPI::MPI_C hdf5_mpich)  # needed for my local computer
       list(APPEND _libraries MPI::MPI_C HDF5::HDF5)
   else()
       list(APPEND _libraries MPI::MPI_C)
   endif()
   include_directories(${HDF5_C_INCLUDE_DIRS})
   if (HDF5_IS_PARALLEL)   # this should be required - needs a better check
        Message(STATUS "Parallel HDF5 supported")
   else()
        Message(STATUS "No parallel HDF5 Support")
   endif()
endif()

# check for FFTW libraries. Cray implementation are not found but have the environmental variable FFTW_ROOT defined.
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:$ENV{FFTW_ROOT}/lib/pkgconfig/")
pkg_check_modules(FFTW QUIET fftw3 IMPORTED_TARGET)
if (TARGET PkgConfig::FFTW)
    message(STATUS "FFTW found")
    list(APPEND _libraries PkgConfig::FFTW)
    link_libraries(PkgConfig::FFTW)
else()
        message(STATUS "FFTW not found")
        message(STATUS "Excluding FFTW specific source code")
endif()


if(USE_DPI)
   set(SRC_DPI src/Main/RegPlugin.cpp src/Core/DiagnosticHook.cpp src/Core/DiagnosticHookBeam.cpp src/Core/DiagnosticHookLI.cpp)
endif()


add_custom_command(
    OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
    ALL
    COMMAND
    ${CMAKE_COMMAND} -D TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/git-hash.cmake
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
    get_git_hash
    ALL
    DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/include/version.h
   
)


include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


# some status print out before compilation
message(STATUS "Build Info:")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
MESSAGE(STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS} )
MESSAGE(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )
Message(STATUS "Used libraries: ${_libraries}")

# objects for the main part - maybe needs a better name
add_library(genesis13 STATIC
        src/Util/BesselJ.cpp
        src/Util/GaussHermite.cpp
        src/Util/Hammerslay.cpp
        src/Util/Inverfc.cpp
        src/Util/RandomU.cpp
	    src/Util/SimpleHandshake.cpp
        src/Util/Sorting.cpp
        src/Util/StringProcessing.cpp
        src/Util/SemaFile.cpp
        src/Util/FFTObj.cpp
        src/Lattice/AlterLattice.cpp
        src/Lattice/Lattice.cpp
        src/Lattice/LatticeElements.cpp
        src/Lattice/LatticeParser.cpp
        src/Lattice/Optics.cpp
        src/Core/Beam.cpp
        src/Core/BeamSolver.cpp
        src/Core/Collective.cpp
        src/Core/Control.cpp
        src/Core/EFieldSolver.cpp
        src/Core/Field.cpp
        src/Core/FieldSolverADI.cpp
        src/Core/FieldSolverFFT.cpp
        src/Core/Gencore.cpp
        src/Core/Incoherent.cpp
        src/Core/TrackBeam.cpp
        src/Core/Undulator.cpp
        src/Core/Diagnostic.cpp
        src/Core/DiagnosticUser.cpp
        src/IO/HDF5base.cpp
        src/IO/Output.cpp
        src/IO/readBeamHDF5.cpp
        src/IO/readFieldHDF5.cpp
        src/IO/SDDSBeam.cpp
        src/IO/writeBeamHDF5.cpp
        src/IO/writeFieldHDF5.cpp
        src/IO/writeWakeHDF5.cpp
        src/IO/readMapHDF5.cpp
        src/Loading/ImportBeam.cpp
        src/Loading/ImportField.cpp
        src/Loading/LoadBeam.cpp
        src/Loading/LoadField.cpp
        src/Loading/GenProfile.cpp
        src/Loading/QuietLoading.cpp
        src/Loading/Series.cpp
        src/Loading/ShotNoise.cpp
        src/Loading/SeriesManager.cpp
        src/Loading/SeriesParser.cpp
        src/Loading/Sequence.cpp
        src/Loading/ImportTransformation.cpp
        src/Main/AlterSetup.cpp
        src/Main/Dump.cpp
        src/Main/AlterBeam.cpp
        src/Main/EField.cpp
        src/Main/FieldManipulator.cpp
        src/Main/GenMain.cpp
        src/Main/mainwrap.cpp
        src/Main/Parser.cpp
        src/Main/Setup.cpp
        src/Main/SponRad.cpp
        src/Main/GenTime.cpp
        src/Main/Track.cpp
        src/Main/Wake.cpp
        ${SRC_DPI}
)



target_include_directories(genesis13
	PRIVATE
	${CMAKE_CURRENT_BINARY_DIR}/include
)


# add optional macro definitions
if (TARGET PkgConfig::FFTW)
    message(STATUS "Defining the macro FFTW in the source code")
    target_compile_definitions(genesis13 PUBLIC FFTW=1)  # changes class layouts in the headers
endif()

if(USE_DPI)
   target_compile_definitions(genesis13 PUBLIC USE_DPI)  # changes class layouts in the headers
endif()


add_dependencies(genesis13 get_git_hash)

# plain-loop SIMD: the particle kernels are auto-vectorized (SimdMath.h)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fopenmp-simd HAVE_FOPENMP_SIMD)
check_cxx_compiler_flag(-fno-math-errno HAVE_FNO_MATH_ERRNO)
if(HAVE_FOPENMP_SIMD)
    target_compile_options(genesis13 PUBLIC -fopenmp-simd)
endif()
if(HAVE_FNO_MATH_ERRNO)
    target_compile_options(genesis13 PUBLIC -fno-math-errno)
endif()
# The ADI field solver reproduces the original scalar solver bit for bit,
# which needs unfused multiply-adds and the full complex products. G4_ADI_FAST
# allows fused multiply-adds and drops the products with the zero real parts
# of the purely imaginary coefficients: last-ulp differences, fewer instructions.
option(G4_ADI_FAST "ADI field solver: allow last-ulp differences for speed" OFF)
if(G4_ADI_FAST)
    set_source_files_properties(src/Core/FieldSolverADI.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=fast" COMPILE_DEFINITIONS G4_ADI_FAST)
else()
    set_source_files_properties(src/Core/FieldSolverADI.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# vec-audit: recompile the files with marked SIMD loops with the compiler's
# vectorization remarks and fail if a marked loop stayed scalar
add_custom_target(vec-audit
    COMMAND python3 ${CMAKE_SOURCE_DIR}/misc/vec-audit.py ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/src/Core/BeamSolver.cpp
        ${CMAKE_SOURCE_DIR}/src/Core/TrackBeam.cpp
        ${CMAKE_SOURCE_DIR}/src/Core/Field.cpp
        ${CMAKE_SOURCE_DIR}/src/Core/Diagnostic.cpp
        ${CMAKE_SOURCE_DIR}/src/Core/DiagnosticUser.cpp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM)

add_executable(genesis4 src/Main/mainwrap.cpp)

target_link_libraries(genesis4 genesis13 stdc++ ${_libraries})

include(CTest)
if(BUILD_TESTING)
    add_executable(lanes_equivalence test/lanes_equivalence.cpp)
    target_link_libraries(lanes_equivalence genesis13 stdc++ ${_libraries})
    add_test(NAME lanes_equivalence COMMAND lanes_equivalence)
endif()


# Compilation of plugin testbench only when plugin system is requested
#if(USE_DPI)
#   add_subdirectory(misc/plugin_testbench)
#endif()


# Test suite. The cases themselves are shell scripts under tests/, so they can
# also be run without a build tree; registering them here is what allows a build
# or a packaging recipe to run them with ctest.
option(BUILD_TESTING "Register the test suite with CTest" ON)
if(BUILD_TESTING)
   enable_testing()
   add_subdirectory(tests)
endif()
