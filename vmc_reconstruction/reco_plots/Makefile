# ------------------------------------------------------------------------------------------------------
#				Set the names of the resulting binary codes
# ------------------------------------------------------------------------------------------------------
XERUSPATH = ./xerus

# Name of the test executable
NAME = Reconstruct


# ------------------------------------------------------------------------------------------------------
#				Register source files for the xerus library      
# ------------------------------------------------------------------------------------------------------

# Register source files



# ------------------------------------------------------------------------------------------------------
#				Load the configurations provided by the user  
# ------------------------------------------------------------------------------------------------------
include $(XERUSPATH)/config.mk


# ------------------------------------------------------------------------------------------------------
#		  			Setup compiler options                
# ------------------------------------------------------------------------------------------------------

# SUGGEST_ATTRIBUTES = TRUE 		# Tell the compiler to suggest attributes

# LOGGING += -D BLAS_ANALYSIS		# Enable BLAS/LAPACK analysis 
  
OTHER += -I $(XERUSPATH)/include	# Add search Path for header files
# OTHER += -std=c++17			# Use old C++11 standard
OTHER += -Wno-ctor-dtor-privacy
OTHER += -Wno-switch-default


# ------------------------------------------------------------------------------------------------------
#					Setup general compiler options      
# ------------------------------------------------------------------------------------------------------
include $(XERUSPATH)/makeIncludes/general.mk
include $(XERUSPATH)/makeIncludes/warnings.mk
include $(XERUSPATH)/makeIncludes/optimization.mk

FLAGS = $(strip $(COMPATIBILITY) $(WARNINGS) $(OPTIMIZE) $(LOGGING) $(DEBUG) $(ADDITIONAL_INCLUDE) $(OTHER))
MINIMAL_DEPS = Makefile $(XERUSPATH)/makeIncludes/general.mk $(XERUSPATH)/makeIncludes/warnings.mk $(XERUSPATH)/makeIncludes/optimization.mk $(XERUSPATH)/config.mk $(XERUSPATH)/build/libxerus.a
# ------------------------------------------------------------------------------------------------------
#					Make Rules     
# ------------------------------------------------------------------------------------------------------
$(NAME): $(MINIMAL_DEPS) $(wildcard *.h) $(wildcard *.cpp)
	$(CXX) $(FLAGS) -g -I $(XERUSPATH)/include/ main.cpp $(XERUSPATH)/build/libxerus.a $(XERUSPATH)/build/libxerus_misc.a -lboost_filesystem -lboost_system $(SUITESPARSE) $(LAPACK_LIBRARIES) $(BLAS_LIBRARIES) -lbfd -liberty -lz -ldl -o $(NAME)

