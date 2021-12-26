%module Ising
%{
#include "Native/lattice.hpp"

%}



%include "Native/lattice.hpp"
%include "typemaps.i"
%include "std_vector.i"
%include "std_string.i"
%include "std_array.i"



%template(IntVector) std::vector<int32_t>;