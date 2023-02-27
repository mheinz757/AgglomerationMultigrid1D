include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;
p = 1;
pAgg = 0;

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

    # agglomerate once
    agg = Vector{Vector{Int64}}(undef, div(n, 2));
    for i in 1:length(agg)
        agg[i] = (2*i-1):(2*i);
    end
    aggMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, dgMesh );

    # create operators and rhs
    G, D, C = aggmg.dg_flux_operators( aggMesh, dgMesh, bdCond, CDir );
    f, r = aggmg.dg_flux_rhs( aggMesh, dgMesh, func, bdCond, CDir );

    b = f - D * ( aggMesh.mMassMatrix \ r );
    A = C - D * sp.sparse( Matrix( aggMesh.mMassMatrix ) \ G );

    # solve problem
    uDg = A \ b;

    # convert modal solution into something we can plot
    x = zeros( 2 * length(aggMesh.mElements) );
    uDgPlot = zeros( length(x) );
    for (i, el) in enumerate( aggMesh.mElements )
        # get x coordinates of boundary
        x[ (2*i-1):2*i ] = el.mBoundingBox;
        basisFunVal = aggmg.evaluate_local_modal_basis_fun( aggMesh.mP, el.mBoundingBox, 
            x[ (2*i-1):2*i ] );

        # get solution value at those x coordinates
        uDgPlot[ 2*i-1 ] = la.dot( uDg[ el.mNodesInd ], basisFunVal[1,:] );
        uDgPlot[ 2*i ] = la.dot( uDg[ el.mNodesInd ], basisFunVal[2,:] );
    end
    perm = sortperm(x);

    # get exact solution
    x2 = range(xin, xout, 101);
    uExact = u_exact.(x2);

    # plot
    ml.put_variable(s1, :x, x[perm]);
    ml.put_variable(s1, :uDgPlot, uDgPlot[perm]);
    ml.put_variable(s1, :x2, x2);
    ml.put_variable(s1, :uExact, uExact);

    ml.eval_string(s1, "figure();");
    ml.eval_string(s1, "hold on;");
    ml.eval_string(s1, "plot(x2, uExact, '-', 'LineWidth', 3)");
    ml.eval_string(s1, "plot(x, uDgPlot, '-o', 'LineWidth', 2, 'MarkerSize', 10)");

    # find l2Error in modal solution
    l2Error = 0.0;
    gaussQuadNodes, gaussQuadWeights = aggmg.gauss_quad( 4*p );
    for el in aggMesh.mElements
        for baseElInd in el.mBaseElementInds
            baseEl = dgMesh.mElements[ baseElInd ];
            baseElGQ = baseEl.mRefMap.(gaussQuadNodes);
            basisGQFunVal = aggmg.evaluate_local_modal_basis_fun( aggMesh.mP, 
                el.mBoundingBox, baseElGQ );

            for (l, xGQ) in enumerate( baseElGQ )
                l2Error += baseEl.mJacobian * gaussQuadWeights[l] * ( u_exact( xGQ ) - 
                    la.dot( uDg[el.mNodesInd], basisGQFunVal[l,:] ) )^2;
            end
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