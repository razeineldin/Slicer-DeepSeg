cmake_minimum_required(VERSION 3.13.4)

project(DeepSeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/razeineldin/Slicer-DeepSeg")
set(EXTENSION_CATEGORY ["Machine Learning", "Segmentation"])
set(EXTENSION_CONTRIBUTORS "Ramy Zeineldin (Reutlingen University, Karlsruhe Institute of Technology), Pauline Weimann (Reutlingen University)")
set(EXTENSION_DESCRIPTION """This modules provides a basic interface for brain tumour segmentation using deep learning-based methods
See more information in <a href="https://github.com/razeineldin/SlicerDeepSeg">module repository</a>.""")
set(EXTENSION_ICONURL "https://github.com/razeineldin/Slicer-DeepSeg/raw/main/Slicer-DeepSeg/DeepSeg/Resources/Icons/DeepSeg.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/razeineldin/Slicer-DeepSeg/raw/main/DeepSeg_Module_UI.png")
set(EXTENSION_DEPENDS "SegmentEditor") # Specified as a list or "NA" if no dependencies
set(EXTENSION_BUILD_SUBDIRECTORY inner-build)

set(SUPERBUILD_TOPLEVEL_PROJECT inner)

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})
mark_as_superbuild(Slicer_DIR)

find_package(Git REQUIRED)
mark_as_superbuild(GIT_EXECUTABLE)

#-----------------------------------------------------------------------------
# SuperBuild setup
option(${EXTENSION_NAME}_SUPERBUILD "Build ${EXTENSION_NAME} and the projects it depends on." ON)
mark_as_advanced(${EXTENSION_NAME}_SUPERBUILD)
if(${EXTENSION_NAME}_SUPERBUILD)
  include("${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild.cmake")
  return()
endif()

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerDeepSeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
set(EXTENSION_CPACK_INSTALL_CMAKE_PROJECTS)
#list(APPEND EXTENSION_CPACK_INSTALL_CMAKE_PROJECTS "${Foo_DIR};Foo;RuntimeLibraries;/")
set(${EXTENSION_NAME}_CPACK_INSTALL_CMAKE_PROJECTS "${EXTENSION_CPACK_INSTALL_CMAKE_PROJECTS}" CACHE STRING "List of external projects to install" FORCE)

#-----------------------------------------------------------------------------
list(APPEND CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${EXTENSION_NAME};ALL;/")
list(APPEND CPACK_INSTALL_CMAKE_PROJECTS "${${EXTENSION_NAME}_CPACK_INSTALL_CMAKE_PROJECTS}")
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
