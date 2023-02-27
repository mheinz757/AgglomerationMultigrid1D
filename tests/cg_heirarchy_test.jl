include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;

n = 128;
maxP = 8;
nCG = 4;

# create model problem
func(x) = cos(x);
u_exact(x) = cos(x);

# create base mesh and set boundary conditions
mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, -sin(xin)), (:dir, cos(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

# create all the cgMeshes
cgMeshes = Vector{aggmg.AbstractMesh}(undef, nCG);
bdConds = Vector{aggmg.BoundaryCondition}(undef, nCG);

tempP = maxP;
for i = 1:nCG
    cgMeshes[i] = aggmg.CgMesh( mesh, tempP );
    bdConds[i] = bdCond;

    global tempP = div( tempP, 2 );
end

A, b = aggmg.cg_stiffness_and_rhs( cgMeshes[1], mesh, func, bdConds[1] );
x0 = 0.0*b;

H = aggmg.MeshHierarchy( cgMeshes, mesh, bdConds, A; nCG = nCG );

maxiter = 100;
tol = 1e-10;
u, iter, res, err = aggmg.multigrid( H, x0, b, maxiter, tol );

println(iter)
println(res)
println(err)