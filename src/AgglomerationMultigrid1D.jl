module AgglomerationMultigrid1D

# import needed packages
import LinearAlgebra as la
import SparseArrays as sp
import SuiteSparse

import Base: *, \
import Base: Array, Matrix, show, similar, size
import LinearAlgebra: ldiv!, lu, mul!
import SparseArrays: sparse

abstract type AbstractElement end
abstract type AbstractAgglomeratedDgElement <: AbstractElement end
abstract type AbstractMesh end
abstract type AbstractSmoother end

include("meshes.jl")
include("boundary_conditions.jl")
include("legendre.jl")
include("gauss_quad.jl")
include("reference_element.jl")
include("block_diagonal.jl")

include("cg_mesh.jl")
include("dg_mesh.jl")
include("agglomerated_dg_mesh.jl")

include("smoother.jl")
include("interpolation.jl")

include("mesh_heirarchy.jl")
include("solvers.jl")

end # end of module