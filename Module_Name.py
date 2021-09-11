
# Retrieve the name of the module that converts the root path of your project to a underscore, i.e. Fw/Types becomes Fw_Types.
get_module_name("${CMAKE_CURRENT_LIST_DIR}")

# Tell the compiler the path to the library header file.
include_directories("${PROJECT_SOURCE_DIR}/libs")

# Find all the libraries contained in the directory you specified.
find_library(LIBQUATERNION Quaternion "${PROJECT_SOURCE_DIR}/libs")
# Note: in my case the LIBQUATERNION equals /home/ThibFrgsGmz/Documents/CODES/fprime-poc/libs/libQuaternion.a

# Specify to the linker the libraries to use when linking the given component and its dependents. 
target_link_libraries("${MODULE_NAME}" ${LIBQUATERNION})
