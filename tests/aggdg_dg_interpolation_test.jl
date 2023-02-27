include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;

p = 1;
pAgg = 0;

n = 16;
CDir = 100.0*n; 

############################################################################################
# test operators on small and large mesh
############################################################################################

func(x) = cos(x);
u_exact(x) = cos(x);

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ (:neu, -sin(xin)), (:dir, cos(xout)) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

baseMesh = aggmg.DgMesh( mesh, p );

# agglomerate
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
aggMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

L = aggmg.aggdg_dg_interpolation( aggMesh, baseMesh );

# get operators on the two levels
baseG, baseD, baseC = aggmg.dg_flux_operators( baseMesh, mesh, bdCond, CDir );
baseA = baseC - baseD * sp.sparse( Matrix(baseMesh.mMassMatrix) \ baseG );

aggG, aggD, aggC = aggmg.dg_flux_operators( aggMesh, baseMesh, bdCond, CDir );
aggA = aggC - aggD * sp.sparse( Matrix(aggMesh.mMassMatrix) \ aggG );

println( "||aggG - L'*baseG*L||: ", la.norm( aggG - L'*baseG*L ) );
println( "||aggD - L'*baseD*L||: ", la.norm( aggD - L'*baseD*L ) );
println( "||aggC - L'*baseC*L||: ", la.norm( aggC - L'*baseC*L ) );
println( "||aggM - L'*baseM*L||: ", la.norm( aggMesh.mMassMatrix - 
    L'*baseMesh.mMassMatrix*L ) );

############################################################################################
# test interpolation of solutions themselves
############################################################################################

u_exact(x) = -x.^4 + 2*x + 3.0
ux_exact(x) = -4*x.^3 + 2.0

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ (:neu, ux_exact(xin)), (:dir, u_exact(xout)) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

baseMesh = aggmg.DgMesh( mesh, p );

# agglomerate
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
aggMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

L = aggmg.aggdg_dg_interpolation( aggMesh, baseMesh );

# get agg solution vector
xAgg = zeros( 2 * length(aggMesh.mElements) );
uAggNodal = zeros( length(xAgg) );
for (i, el) in enumerate( aggMesh.mElements )
    xAgg[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if aggMesh.mP == 0
        uAggNodal[ (2*i-1):2*i ] .= ( u_exact( xAgg[2*i-1] ) + u_exact( xAgg[2*i] ) ) / 2.0;
    elseif aggMesh.mP == 1
        uAggNodal[ (2*i-1):2*i ] = u_exact.( xAgg[ (2*i-1):2*i ] );
    end
end
permAgg = sortperm( xAgg );

uAgg = zeros( aggMesh.mNumNodes );
for (i, el) in enumerate( aggMesh.mElements )
    if aggMesh.mP == 0
        uAgg[el.mNodesInd] .= ( uAggNodal[2*i-1] + uAggNodal[2*i] ) / 2.0;
    elseif aggMesh.mP == 1
        uAgg[el.mNodesInd[1]] = ( uAggNodal[2*i-1] + uAggNodal[2*i] ) / 2.0;
        uAgg[el.mNodesInd[2]] = ( uAggNodal[2*i] - uAggNodal[2*i-1] ) / 2.0;
    end
end

# get base solution vector
xBase = zeros( baseMesh.mNumNodes );
for (i, el) in enumerate( baseMesh.mElements )
    xBase[ el.mNodesInd ] = el.mNodesX;
    xBase[ el.mNodesInd[1] ] += 1e-14;
    xBase[ el.mNodesInd[2] ] -= 1e-14;
end
permBase = sortperm( xBase );
uBase = u_exact.( xBase );

# find interpolated solution
uBaseInterp = L * uAgg;

# calculate errors in solution
l2ErrorReg = 0.0;
l2ErrorInterp = 0.0;

refEl = baseMesh.mRefEl;
gaussQuadNodes = refEl.mGaussQuadNodes;
gaussQuadWeights = refEl.mGaussQuadWeights;

for aggEl in aggMesh.mElements
    for (k, baseElInd) in enumerate( aggEl.mBaseElementInds )
        baseEl = baseMesh.mElements[ baseElInd ];
        aggBasisGQFunVal = aggmg.evaluate_local_modal_basis_fun( aggMesh.mP, 
                aggEl.mBoundingBox, baseEl.mRefMap.(gaussQuadNodes) );

        for (l, xGQ) in enumerate( gaussQuadNodes )
            global l2ErrorReg += baseEl.mJacobian * gaussQuadWeights[l] * 
                ( la.dot( uAgg[aggEl.mNodesInd], aggBasisGQFunVal[l,:] ) - 
                la.dot( uBase[baseEl.mNodesInd], refEl.mBasisGQFunVal[l,:] ) )^2;
            global l2ErrorInterp += baseEl.mJacobian * gaussQuadWeights[l] * 
                ( la.dot( uAgg[aggEl.mNodesInd], aggBasisGQFunVal[l,:] ) - 
                la.dot( uBaseInterp[baseEl.mNodesInd], refEl.mBasisGQFunVal[l,:] ) )^2;
        end
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorInterp = sqrt(l2ErrorInterp);

println("\nDifferences in solutions when interpolating:")
println("||uAgg - uBase||: ", l2ErrorReg)
println("||uAgg - uBaseInterp||: ", l2ErrorInterp)

# plot
s1 = ml.get_default_msession();

ml.put_variable(s1, :xAgg, xAgg[permAgg]);
ml.put_variable(s1, :uAgg, uAggNodal[permAgg]);
ml.put_variable(s1, :xBase, xBase[permBase]);
ml.put_variable(s1, :uBase, uBase[permBase]);
ml.put_variable(s1, :uBaseInterp, uBaseInterp[permBase]);

ml.eval_string(s1, "figure();");
ml.eval_string(s1, "hold on;");
ml.eval_string(s1, "plot(xAgg, uAgg, '-s', 'LineWidth', 4, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xBase, uBase, '-o', 'LineWidth', 2, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xBase, uBaseInterp, '-*', 'LineWidth', 2, 'MarkerSize', 10)");

############################################################################################
# test restriction too
############################################################################################

# u_exact(x) = -x.^2 + 2.0*x + 3.0;
# ux_exact(x) = -2.0*x + 2.0;
u_exact(x) = 2.0*x + 3.0;
ux_exact(x) = 2.0;

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ (:neu, ux_exact(xin)), (:dir, u_exact(xout)) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

baseMesh = aggmg.DgMesh( mesh, p );

# agglomerate
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
aggMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

L = aggmg.aggdg_dg_interpolation( aggMesh, baseMesh );

# get agg solution vector
xAgg = zeros( 2 * length(aggMesh.mElements) );
uAggNodal = zeros( length(xAgg) );
for (i, el) in enumerate( aggMesh.mElements )
    xAgg[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if aggMesh.mP == 0
        uAggNodal[ (2*i-1):2*i ] .= ( u_exact( xAgg[2*i-1] ) + u_exact( xAgg[2*i] ) ) / 2.0;
    elseif aggMesh.mP == 1
        uAggNodal[ (2*i-1):2*i ] = u_exact.( xAgg[ (2*i-1):2*i ] );
    end
end
permAgg = sortperm( xAgg );

uAgg = zeros( aggMesh.mNumNodes );
for (i, el) in enumerate( aggMesh.mElements )
    if aggMesh.mP == 0
        uAgg[el.mNodesInd] .= ( uAggNodal[2*i-1] + uAggNodal[2*i] ) / 2.0;
    elseif aggMesh.mP == 1
        uAgg[el.mNodesInd[1]] = ( uAggNodal[2*i-1] + uAggNodal[2*i] ) / 2.0;
        uAgg[el.mNodesInd[2]] = ( uAggNodal[2*i] - uAggNodal[2*i-1] ) / 2.0;
    end
end

# get base solution vector
xBase = zeros( baseMesh.mNumNodes );
for (i, el) in enumerate( baseMesh.mElements )
    xBase[ el.mNodesInd ] = el.mNodesX;
    xBase[ el.mNodesInd[1] ] += 1e-14;
    xBase[ el.mNodesInd[2] ] -= 1e-14;
end
permBase = sortperm( xBase );
uBase = u_exact.( xBase );

# find restricted solution, and nodal form of it
uAggRestr = aggMesh.mMassMatrixLU \ ( L' * (baseMesh.mMassMatrix * uBase) );

uAggRestrNodal = zeros( length(xAgg) );
for (i, el) in enumerate( aggMesh.mElements )
    basisFunVal = aggmg.evaluate_local_modal_basis_fun( aggMesh.mP, el.mBoundingBox, 
        xAgg[ (2*i-1):2*i ] );
    uAggRestrNodal[ 2*i-1 ] = la.dot( uAggRestr[ el.mNodesInd ], basisFunVal[1,:] );
    uAggRestrNodal[ 2*i ] = la.dot( uAggRestr[ el.mNodesInd ], basisFunVal[2,:] );
end

# calculate errors in solution
l2ErrorReg = 0.0;
l2ErrorRestr = 0.0;

refEl = baseMesh.mRefEl;
gaussQuadNodes = refEl.mGaussQuadNodes;
gaussQuadWeights = refEl.mGaussQuadWeights;

for aggEl in aggMesh.mElements
    for (k, baseElInd) in enumerate( aggEl.mBaseElementInds )
        baseEl = baseMesh.mElements[ baseElInd ];
        aggBasisGQFunVal = aggmg.evaluate_local_modal_basis_fun( aggMesh.mP, 
                aggEl.mBoundingBox, baseEl.mRefMap.(gaussQuadNodes) );

        for (l, xGQ) in enumerate( gaussQuadNodes )
            global l2ErrorReg += baseEl.mJacobian * gaussQuadWeights[l] * 
                ( la.dot( uAgg[aggEl.mNodesInd], aggBasisGQFunVal[l,:] ) - 
                la.dot( uBase[baseEl.mNodesInd], refEl.mBasisGQFunVal[l,:] ) )^2;
            global l2ErrorRestr += baseEl.mJacobian * gaussQuadWeights[l] * 
                ( la.dot( uAggRestr[aggEl.mNodesInd], aggBasisGQFunVal[l,:] ) - 
                la.dot( uBase[baseEl.mNodesInd], refEl.mBasisGQFunVal[l,:] ) )^2;
        end
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorRestr = sqrt(l2ErrorRestr);

println("\nDifferences in solutions when restricting:")
println("||uAgg - uBase||: ", l2ErrorReg)
println("||uAggRestr - uBase||: ", l2ErrorRestr)

# plot
s1 = ml.get_default_msession();

ml.put_variable(s1, :xAgg, xAgg[permAgg]);
ml.put_variable(s1, :uAgg, uAggNodal[permAgg]);
ml.put_variable(s1, :uAggRestr, uAggRestrNodal[permAgg]);
ml.put_variable(s1, :xBase, xBase[permBase]);
ml.put_variable(s1, :uBase, uBase[permBase]);


ml.eval_string(s1, "figure();");
ml.eval_string(s1, "hold on;");
ml.eval_string(s1, "plot(xAgg, uAgg, '-s', 'LineWidth', 4, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xAgg, uAggRestr, '-*', 'LineWidth', 2, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xBase, uBase, '-o', 'LineWidth', 2, 'MarkerSize', 10)");

############################################################################################
# test interpolation of discontinuous solutions
############################################################################################

function u_exact(x)
    if x < 0.5
        res = -x.^2 + 2*x + 3.0;
    else
        res = -x.^2 - 2*x + 3.0;
    end

    return res;
end
function ux_exact(x)
    if x < 0.5
        res = -2*x.^1 + 2.0;
    else
        res = -2*x.^1 - 2.0;
    end

    return res;
end

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ (:neu, ux_exact(xin)), (:dir, u_exact(xout)) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

baseMesh = aggmg.DgMesh( mesh, p );

# agglomerate
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
aggMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

L = aggmg.aggdg_dg_interpolation( aggMesh, baseMesh );

# get agg solution vector
xAgg = zeros( 2 * length(aggMesh.mElements) );
uAggNodal = zeros( length(xAgg) );
for (i, el) in enumerate( aggMesh.mElements )
    xAgg[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if aggMesh.mP == 0
        uAggNodal[ (2*i-1):2*i ] .= ( u_exact( xAgg[2*i-1] ) + u_exact( xAgg[2*i] ) ) / 2.0;
    elseif aggMesh.mP == 1
        uAggNodal[ (2*i-1):2*i ] = u_exact.( xAgg[ (2*i-1):2*i ] );
    end
end
permAgg = sortperm( xAgg );

uAgg = zeros( aggMesh.mNumNodes );
for (i, el) in enumerate( aggMesh.mElements )
    if aggMesh.mP == 0
        uAgg[el.mNodesInd] .= ( uAggNodal[2*i-1] + uAggNodal[2*i] ) / 2.0;
    elseif aggMesh.mP == 1
        uAgg[el.mNodesInd[1]] = ( uAggNodal[2*i-1] + uAggNodal[2*i] ) / 2.0;
        uAgg[el.mNodesInd[2]] = ( uAggNodal[2*i] - uAggNodal[2*i-1] ) / 2.0;
    end
end

# get base solution vector
xBase = zeros( baseMesh.mNumNodes );
for (i, el) in enumerate( baseMesh.mElements )
    xBase[ el.mNodesInd ] = el.mNodesX;
    xBase[ el.mNodesInd[1] ] += 1e-14;
    xBase[ el.mNodesInd[2] ] -= 1e-14;
end
permBase = sortperm( xBase );
uBase = u_exact.( xBase );

# find interpolated solution
uBaseInterp = L * uAgg;

# calculate errors in solution
l2ErrorReg = 0.0;
l2ErrorInterp = 0.0;

refEl = baseMesh.mRefEl;
gaussQuadNodes = refEl.mGaussQuadNodes;
gaussQuadWeights = refEl.mGaussQuadWeights;

for aggEl in aggMesh.mElements
    for (k, baseElInd) in enumerate( aggEl.mBaseElementInds )
        baseEl = baseMesh.mElements[ baseElInd ];
        aggBasisGQFunVal = aggmg.evaluate_local_modal_basis_fun( aggMesh.mP, 
                aggEl.mBoundingBox, baseEl.mRefMap.(gaussQuadNodes) );

        for (l, xGQ) in enumerate( gaussQuadNodes )
            global l2ErrorReg += baseEl.mJacobian * gaussQuadWeights[l] * 
                ( la.dot( uAgg[aggEl.mNodesInd], aggBasisGQFunVal[l,:] ) - 
                la.dot( uBase[baseEl.mNodesInd], refEl.mBasisGQFunVal[l,:] ) )^2;
            global l2ErrorInterp += baseEl.mJacobian * gaussQuadWeights[l] * 
                ( la.dot( uAgg[aggEl.mNodesInd], aggBasisGQFunVal[l,:] ) - 
                la.dot( uBaseInterp[baseEl.mNodesInd], refEl.mBasisGQFunVal[l,:] ) )^2;
        end
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorInterp = sqrt(l2ErrorInterp);

println("\nDifferences in discontinuous solutions when interpolating:")
println("||uAgg - uBase||: ", l2ErrorReg)
println("||uAgg - uBaseInterp||: ", l2ErrorInterp)

# plot
s1 = ml.get_default_msession();

ml.put_variable(s1, :xAgg, xAgg[permAgg]);
ml.put_variable(s1, :uAgg, uAggNodal[permAgg]);
ml.put_variable(s1, :xBase, xBase[permBase]);
ml.put_variable(s1, :uBase, uBase[permBase]);
ml.put_variable(s1, :uBaseInterp, uBaseInterp[permBase]);

ml.eval_string(s1, "figure();");
ml.eval_string(s1, "hold on;");
ml.eval_string(s1, "plot(xAgg, uAgg, '-s', 'LineWidth', 4, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xBase, uBase, '-o', 'LineWidth', 2, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xBase, uBaseInterp, '-*', 'LineWidth', 2, 'MarkerSize', 10)");