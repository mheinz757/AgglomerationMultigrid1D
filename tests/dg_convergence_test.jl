include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;
p = 3;

func(x) = cos(x);
u_exact(x) = cos(x);

nVec = [4, 8, 16, 32, 64, 128, 256];
errorVec = Vector{Float64}(undef, length(nVec));

s1 = ml.get_default_msession();

for (i, n) in enumerate(nVec)
    CDir = 1.0*n;

    mesh = create_uniform_mesh( n, xin, xout );
    dgMesh = aggmg.DgMesh( mesh, p );

    mBdCond = [(:dir, cos(xin)), (:neu, -sin(xout))];
    bdCond = set_boundary!( mesh, xin, xout, mBdCond );

    G, D, C = aggmg.dg_flux_operators( dgMesh, mesh, bdCond, CDir );
    f, r = aggmg.dg_flux_rhs( dgMesh, mesh, func, bdCond, CDir );

    b = f - D * (dgMesh.mMassMatrix \ r);
    A = C - D * sp.sparse(Matrix(dgMesh.mMassMatrix) \ G);

    uDg = A \ b;

    x = zeros( dgMesh.mNumNodes );
    for el in dgMesh.mElements
        x[ el.mNodesInd ] = el.mNodesX;
    end
    perm = sortperm(x);

    x2 = range(xin, xout, 101)
    uExact = u_exact.(x2);
    

    ml.put_variable(s1, :x, x[perm]);
    ml.put_variable(s1, :uDg, uDg[perm]);
    ml.put_variable(s1, :x2, x2);
    ml.put_variable(s1, :uExact, uExact);

    ml.eval_string(s1, "figure();");
    ml.eval_string(s1, "hold on;");
    ml.eval_string(s1, "plot(x2, uExact, '-', 'LineWidth', 3)");
    ml.eval_string(s1, "plot(x, uDg, '-o', 'LineWidth', 2, 'MarkerSize', 10)");

    l2Error = 0.0;
    gaussQuadNodes, gaussQuadWeights = aggmg.gauss_quad( 4*p );
    basisGQFunVal = aggmg.evaluate_nodal_basis_fun( dgMesh.mRefEl.mBasisFunCoeff, 
        gaussQuadNodes);
    for el in dgMesh.mElements
        for (l, xGQ) in enumerate(gaussQuadNodes)
            l2Error += el.mJacobian * gaussQuadWeights[l] * ( u_exact( el.mRefMap(xGQ) ) - 
                la.dot( uDg[el.mNodesInd], basisGQFunVal[l,:] ) )^2;
        end
    end
    l2Error = sqrt(l2Error);

    errorVec[i] = l2Error;
end

logErrorVec = log.(10, errorVec);
logHVec = log.(10, 1.0./nVec);

ml.put_variable(s1, :logErrorVec, logErrorVec);
ml.put_variable(s1, :logHVec, logHVec);

ml.eval_string(s1, "figure();");
ml.eval_string(s1, "hold on;");
ml.eval_string(s1, "plot(logHVec, logErrorVec, '-s', 'LineWidth', 2, 'MarkerSize', 10)");

println((logErrorVec[end]-logErrorVec[1])/(logHVec[end]-logHVec[1]));