include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;

n = 128;
CDir = 1000.0*n;

maxP = 8;
nCG = 4;
nDG = 1;

# create model problem
func(x) = cos(x);
u_exact(x) = cos(x);

# create base mesh and set boundary conditions
mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, -sin(xin)), (:dir, cos(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

# create all the meshes
mMeshes = Vector{aggmg.AbstractMesh}(undef, nCG+nDG);
bdConds = Vector{aggmg.BoundaryCondition}(undef, nCG+nDG);

tempP = maxP;
for i = 1:nCG
    mMeshes[i] = aggmg.CgMesh( mesh, tempP );
    bdConds[i] = bdCond;

    global tempP = div( tempP, 2 );
end

# tempP = 1;
for i = 1:nDG
    mMeshes[nCG+i] = aggmg.DgMesh( mesh, tempP );
    bdConds[nCG+i] = bdCond;

    global tempP = div( tempP, 2 );
end

A, b = aggmg.cg_stiffness_and_rhs( mMeshes[1], mesh, func, bdConds[1] );
x0 = 0.0*b;

H = aggmg.MeshHierarchy( mMeshes, mesh, bdConds, A; nCG = nCG, nDG = nDG, CDir = CDir );

maxiter = 100;
tol = 1e-10;
u, iter, res, err = aggmg.multigrid( H, x0, b, maxiter, tol );

println(iter)
println(res)
println(err)