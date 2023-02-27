include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;

# n = 512;
# CDir = 1000.0*n;
nVec = 2 .^ (3:9);
iterVec = zeros( Int64, length(nVec) );

maxP = 8;
pAgg = 1;

nCG = 4;
nDG = 0;

# create model problem
func(x) = cos(x);
u_exact(x) = cos(x);
ux_exact(x) = -sin(x);
# func(x) = exp(-x);
# u_exact(x) = -exp(-x) - 1.0*x + 3.0;
# ux_exact(x) = exp(-x) - 1.0;

for (k, n) in enumerate( nVec )
    # basic parameters 
    CDir = 1000.0*n;
    # nAgg = Int64( log2(n) );
    nAgg = Int64( log2(n) ) - 1;

    # create base mesh and set boundary conditions
    mesh = create_uniform_mesh( n, xin, xout );
    # mBdCond = [(:neu, -sin(xin)), (:dir, cos(xout))];
    mBdCond = [(:neu, ux_exact(xin)), (:dir, u_exact(xout))];
    bdCond = set_boundary!( mesh, xin, xout, mBdCond );

    # create all the meshes
    mMeshes = Vector{aggmg.AbstractMesh}(undef, nCG+nDG+nAgg);
    bdConds = Vector{aggmg.BoundaryCondition}(undef, nCG+nDG+nAgg);

    tempP = maxP;
    for i = 1:nCG
        mMeshes[i] = aggmg.CgMesh( mesh, tempP );
        bdConds[i] = bdCond;

        tempP = div( tempP, 2 );
    end

    # tempP = 1;
    for i = 1:nDG
        mMeshes[nCG+i] = aggmg.DgMesh( mesh, tempP );
        bdConds[nCG+i] = bdCond;

        tempP = div( tempP, 2 );
    end

    # tempN = div( n, 2 );
    tempN = div( n, 4 );
    for i = 1:nAgg
        agg = Vector{Vector{Int64}}( undef, tempN );
        if i == 1
            for j in eachindex(agg)
                # agg[j] = (2*j-1):(2*j);
                agg[j] = (4*j-3):(4*j);
            end
        else 
            for j in eachindex(agg)
                agg[j] = (2*j-1):(2*j);
            end
        end

        if i == 1
            mMeshes[nCG+nDG+i] = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, mMeshes[1] );
        else
            mMeshes[nCG+nDG+i] = aggmg.AgglomeratedDgMeshN( pAgg, agg, mMeshes[nCG+nDG+i-1], 
                mMeshes[1] );
        end
        bdConds[nCG+nDG+i] = bdCond;

        tempN = div( tempN, 2);
    end

    A, b = aggmg.cg_stiffness_and_rhs( mMeshes[1], mesh, func, bdConds[1] );
    x0 = 0.0*b;

    H = aggmg.MeshHierarchy( mMeshes, mesh, bdConds, A; nCG = nCG, nDG = nDG, nAgg = nAgg, 
        CDir = CDir );

    maxiter = 100;
    tol = 1e-10;
    u, iter, res, err = aggmg.multigrid( H, x0, b, maxiter, tol );

    println(iter)
    # println(res)
    # println(err)

    iterVec[k] = iter;
end