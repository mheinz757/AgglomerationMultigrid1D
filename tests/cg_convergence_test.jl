include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la

# basic parameters
xin = 0.0;
xout = 1.0;
p = 3;

func(x) = cos(x);
u_exact(x) = cos(x);

nVec = [4, 8, 16, 32, 64];
errorVec = Vector{Float64}(undef, length(nVec));

s1 = ml.get_default_msession();

for (i, n) in enumerate(nVec)
    mesh = create_uniform_mesh( n, xin, xout );
    cgMesh = aggmg.CgMesh( mesh, p );

    mBdCond = [(:neu, -sin(xin)), (:dir, cos(xout))];
    bdCond = set_boundary!( mesh, xin, xout, mBdCond );


    A = aggmg.cg_stiffness( cgMesh, bdCond );
    A2, f2 = aggmg.cg_stiffness_and_rhs( cgMesh, mesh, func, bdCond );
    f = aggmg.cg_rhs( cgMesh, mesh, func, bdCond );

    # println(la.norm(A - A2));
    # println(la.norm(f - f2));

    uCg = A \ f;

    x = zeros( cgMesh.mNumNodes );
    for el in cgMesh.mElements
        x[ el.mNodesInd ] = el.mNodesX;
    end
    perm = sortperm(x);

    x2 = range(xin, xout, 101)
    uExact = u_exact.(x2);
    

    ml.put_variable(s1, :x, x[perm]);
    ml.put_variable(s1, :uCg, uCg[perm]);
    ml.put_variable(s1, :x2, x2);
    ml.put_variable(s1, :uExact, uExact);

    ml.eval_string(s1, "figure();");
    ml.eval_string(s1, "hold on;");
    ml.eval_string(s1, "plot(x2, uExact, '-', 'LineWidth', 3)");
    ml.eval_string(s1, "plot(x, uCg, '-o', 'LineWidth', 2, 'MarkerSize', 10)");

    l2Error = 0.0;
    gaussQuadNodes, gaussQuadWeights = aggmg.gauss_quad( 4*p );
    basisGQFunVal = aggmg.evaluate_nodal_basis_fun( cgMesh.mRefEl.mBasisFunCoeff, 
        gaussQuadNodes);
    for el in cgMesh.mElements
        for (l, xGQ) in enumerate(gaussQuadNodes)
            l2Error += el.mJacobian * gaussQuadWeights[l] * ( u_exact( el.mRefMap(xGQ) ) - 
                la.dot( uCg[el.mNodesInd], basisGQFunVal[l,:] ) )^2;
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