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
nDG = 4;

# create model problem
func(x) = cos(x);
u_exact(x) = cos(x);

# create base mesh and set boundary conditions
mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, -sin(xin)), (:dir, cos(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

# create all the dgMeshes
dgMeshes = Vector{aggmg.AbstractMesh}(undef, nDG);
bdConds = Vector{aggmg.BoundaryCondition}(undef, nDG);

tempP = maxP;
for i = 1:nDG
    dgMeshes[i] = aggmg.DgMesh( mesh, tempP );
    bdConds[i] = bdCond;

    global tempP = div( tempP, 2 );
end

G, D, C = aggmg.dg_flux_operators( dgMeshes[1], mesh, bdConds[1], CDir );
A = C - D * sp.sparse(Matrix(dgMeshes[1].mMassMatrix) \ G);

f, r = aggmg.dg_flux_rhs( dgMeshes[1], mesh, func, bdConds[1], CDir );
b = f - D * (dgMeshes[1].mMassMatrix \ r);

x0 = 0.0*b;

H = aggmg.MeshHierarchy( dgMeshes, bdConds, A, G, D, C; nDG = nDG );

maxiter = 200;
tol = 1e-10;
u, iter, res, err = aggmg.multigrid( H, x0, b, maxiter, tol );

println(iter)
println(res)
println(err)