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

# agglomerate twice
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
fineMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

agg2 = Vector{Vector{Int64}}( undef, div(n, 4) );
for i in 1:length(agg2)
    agg2[i] = (2*i-1):(2*i);
end
coarseMesh = aggmg.AgglomeratedDgMeshN( pAgg, agg2, fineMesh, baseMesh );

L = aggmg.aggdg_aggdg_interpolation( coarseMesh, fineMesh, baseMesh );

# also make an agglomeration that does two agglomerations at once
agg3 = Vector{Vector{Int64}}( undef, div(n, 4) );
for i in 1:length(agg3)
    agg3[i] = (4*i-3):(4*i);
end
coarseMesh2 = aggmg.AgglomeratedDgMesh1( pAgg, agg3, mesh, baseMesh );

# get operators on the two levels
coarseG, coarseD, coarseC = aggmg.dg_flux_operators( coarseMesh2, baseMesh, bdCond, CDir );
coarseA = coarseC - coarseD * sp.sparse( Matrix(coarseMesh2.mMassMatrix) \ coarseG );

fineG, fineD, fineC = aggmg.dg_flux_operators( fineMesh, baseMesh, bdCond, CDir );
fineA = fineC - fineD * sp.sparse( Matrix(fineMesh.mMassMatrix) \ fineG );

println( "||coarseG - L'*fineG*L||: ", la.norm( coarseG - L'*fineG*L ) );
println( "||coarseD - L'*fineD*L||: ", la.norm( coarseD - L'*fineD*L ) );
println( "||coarseC - L'*fineC*L||: ", la.norm( coarseC - L'*fineC*L ) );
println( "||coarseM - L'*fineM*L||: ", la.norm( coarseMesh2.mMassMatrix - 
    L'*fineMesh.mMassMatrix*L ) );

############################################################################################
# test interpolation of solutions themselves
############################################################################################

u_exact(x) = -x.^4 + 2*x + 3.0
ux_exact(x) = -4*x.^3 + 2.0

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ (:neu, ux_exact(xin)), (:dir, u_exact(xout)) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

baseMesh = aggmg.DgMesh( mesh, p );

# agglomerate twice
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
fineMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

agg2 = Vector{Vector{Int64}}( undef, div(n, 4) );
for i in 1:length(agg2)
    agg2[i] = (2*i-1):(2*i);
end
coarseMesh = aggmg.AgglomeratedDgMeshN( pAgg, agg2, fineMesh, baseMesh );

L = aggmg.aggdg_aggdg_interpolation( coarseMesh, fineMesh, baseMesh );

# get coarse solution vector
xCoarse = zeros( 2 * length(coarseMesh.mElements) );
uCoarseNodal = zeros( length(xCoarse) );
for (i, el) in enumerate( coarseMesh.mElements )
    xCoarse[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if coarseMesh.mP == 0
        uCoarseNodal[ (2*i-1):2*i ] .= 
            ( u_exact( xCoarse[2*i-1] ) + u_exact( xCoarse[2*i] ) ) / 2.0;
    elseif coarseMesh.mP == 1
        uCoarseNodal[ (2*i-1):2*i ] = u_exact.( xCoarse[ (2*i-1):2*i ] );
    end
end
permCoarse = sortperm( xCoarse );

uCoarse = zeros( coarseMesh.mNumNodes );
for (i, el) in enumerate( coarseMesh.mElements )
    if coarseMesh.mP == 0
        uCoarse[el.mNodesInd] .= ( uCoarseNodal[2*i-1] + uCoarseNodal[2*i] ) / 2.0;
    elseif coarseMesh.mP == 1
        uCoarse[el.mNodesInd[1]] = ( uCoarseNodal[2*i-1] + uCoarseNodal[2*i] ) / 2.0;
        uCoarse[el.mNodesInd[2]] = ( uCoarseNodal[2*i] - uCoarseNodal[2*i-1] ) / 2.0;
    end
end

# get fine solution vector
xFine = zeros( 2 * length(fineMesh.mElements) );
uFineNodal = zeros( length(xFine) );
for (i, el) in enumerate( fineMesh.mElements )
    xFine[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if fineMesh.mP == 0
        uFineNodal[ (2*i-1):2*i ] .= 
            ( u_exact( xFine[2*i-1] ) + u_exact( xFine[2*i] ) ) / 2.0;
    elseif fineMesh.mP == 1
        uFineNodal[ (2*i-1):2*i ] = u_exact.( xFine[ (2*i-1):2*i ] );
    end
end
permFine = sortperm( xFine );

uFine = zeros( fineMesh.mNumNodes );
for (i, el) in enumerate( fineMesh.mElements )
    if fineMesh.mP == 0
        uFine[el.mNodesInd] .= ( uFineNodal[2*i-1] + uFineNodal[2*i] ) / 2.0;
    elseif fineMesh.mP == 1
        uFine[el.mNodesInd[1]] = ( uFineNodal[2*i-1] + uFineNodal[2*i] ) / 2.0;
        uFine[el.mNodesInd[2]] = ( uFineNodal[2*i] - uFineNodal[2*i-1] ) / 2.0;
    end
end

# find interpolated solution, and nodal form of it
uFineInterp = L * uCoarse;

uFineInterpNodal = zeros( length(xFine) );
for (i, el) in enumerate( fineMesh.mElements )
    basisFunVal = aggmg.evaluate_local_modal_basis_fun( fineMesh.mP, el.mBoundingBox, 
        xFine[ (2*i-1):2*i ] );
    uFineInterpNodal[ 2*i-1 ] = la.dot( uFineInterp[ el.mNodesInd ], basisFunVal[1,:] );
    uFineInterpNodal[ 2*i ] = la.dot( uFineInterp[ el.mNodesInd ], basisFunVal[2,:] );
end

# calculate errors in solution
l2ErrorReg = 0.0;
l2ErrorInterp = 0.0;

for coarseEl in coarseMesh.mElements
    count = 0;
    for subElInd in coarseEl.mSubAggElementInds
        fineEl = fineMesh.mElements[ subElInd ];
        for (k, baseElInd) in enumerate( fineEl.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            for (l, xGQ) in enumerate( fineMesh.mGaussQuadNodes )
                global l2ErrorReg += baseEl.mJacobian * fineMesh.mGaussQuadWeights[l] * 
                    ( la.dot( uCoarse[coarseEl.mNodesInd], coarseEl.mBasisGQFunVal[count+k][l,:] ) - 
                    la.dot( uFine[fineEl.mNodesInd], fineEl.mBasisGQFunVal[k][l,:] ) )^2;
                global l2ErrorInterp += baseEl.mJacobian * fineMesh.mGaussQuadWeights[l] * 
                    ( la.dot( uCoarse[coarseEl.mNodesInd], coarseEl.mBasisGQFunVal[count+k][l,:] ) - 
                    la.dot( uFineInterp[fineEl.mNodesInd], fineEl.mBasisGQFunVal[k][l,:] ) )^2;
            end
        end
        count += length( fineEl.mBaseElementInds );
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorInterp = sqrt(l2ErrorInterp);

println("\nDifferences in solutions when interpolating:")
println("||uCoarse - uFine||: ", l2ErrorReg)
println("||uCoarse - uFineInterp||: ", l2ErrorInterp)

# plot
s1 = ml.get_default_msession();

ml.put_variable(s1, :xCoarse, xCoarse[permCoarse]);
ml.put_variable(s1, :uCoarse, uCoarseNodal[permCoarse]);
ml.put_variable(s1, :xFine, xFine[permFine]);
ml.put_variable(s1, :uFine, uFineNodal[permFine]);
ml.put_variable(s1, :uFineInterp, uFineInterpNodal[permFine]);

ml.eval_string(s1, "figure();");
ml.eval_string(s1, "hold on;");
ml.eval_string(s1, "plot(xCoarse, uCoarse, '-s', 'LineWidth', 4, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xFine, uFine, '-o', 'LineWidth', 2, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xFine, uFineInterp, '-*', 'LineWidth', 2, 'MarkerSize', 10)");

############################################################################################
# test restriction too
############################################################################################

# u_exact(x) = -x.^2 + 2.0*x + 3.0
# ux_exact(x) = -2.0*x + 2.0
u_exact(x) = 2.0*x + 3.0
ux_exact(x) = 2.0

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ (:neu, ux_exact(xin)), (:dir, u_exact(xout)) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

baseMesh = aggmg.DgMesh( mesh, p );

# agglomerate twice
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
fineMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

agg2 = Vector{Vector{Int64}}( undef, div(n, 4) );
for i in 1:length(agg2)
    agg2[i] = (2*i-1):(2*i);
end
coarseMesh = aggmg.AgglomeratedDgMeshN( pAgg, agg2, fineMesh, baseMesh );

L = aggmg.aggdg_aggdg_interpolation( coarseMesh, fineMesh, baseMesh );

# get coarse solution vector
xCoarse = zeros( 2 * length(coarseMesh.mElements) );
uCoarseNodal = zeros( length(xCoarse) );
for (i, el) in enumerate( coarseMesh.mElements )
    xCoarse[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if coarseMesh.mP == 0
        uCoarseNodal[ (2*i-1):2*i ] .= 
            ( u_exact( xCoarse[2*i-1] ) + u_exact( xCoarse[2*i] ) ) / 2.0;
    elseif coarseMesh.mP == 1
        uCoarseNodal[ (2*i-1):2*i ] = u_exact.( xCoarse[ (2*i-1):2*i ] );
    end
end
permCoarse = sortperm( xCoarse );

uCoarse = zeros( coarseMesh.mNumNodes );
for (i, el) in enumerate( coarseMesh.mElements )
    if coarseMesh.mP == 0
        uCoarse[el.mNodesInd] .= ( uCoarseNodal[2*i-1] + uCoarseNodal[2*i] ) / 2.0;
    elseif coarseMesh.mP == 1
        uCoarse[el.mNodesInd[1]] = ( uCoarseNodal[2*i-1] + uCoarseNodal[2*i] ) / 2.0;
        uCoarse[el.mNodesInd[2]] = ( uCoarseNodal[2*i] - uCoarseNodal[2*i-1] ) / 2.0;
    end
end

# get fine solution vector
xFine = zeros( 2 * length(fineMesh.mElements) );
uFineNodal = zeros( length(xFine) );
for (i, el) in enumerate( fineMesh.mElements )
    xFine[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if fineMesh.mP == 0
        uFineNodal[ (2*i-1):2*i ] .= 
            ( u_exact( xFine[2*i-1] ) + u_exact( xFine[2*i] ) ) / 2.0;
    elseif fineMesh.mP == 1
        uFineNodal[ (2*i-1):2*i ] = u_exact.( xFine[ (2*i-1):2*i ] );
    end
end
permFine = sortperm( xFine );

uFine = zeros( fineMesh.mNumNodes );
for (i, el) in enumerate( fineMesh.mElements )
    if fineMesh.mP == 0
        uFine[el.mNodesInd] .= ( uFineNodal[2*i-1] + uFineNodal[2*i] ) / 2.0;
    elseif fineMesh.mP == 1
        uFine[el.mNodesInd[1]] = ( uFineNodal[2*i-1] + uFineNodal[2*i] ) / 2.0;
        uFine[el.mNodesInd[2]] = ( uFineNodal[2*i] - uFineNodal[2*i-1] ) / 2.0;
    end
end

# find restricted solution, and nodal form of it
uCoarseRestr = coarseMesh.mMassMatrixLU \ ( L' * (fineMesh.mMassMatrix * uFine) );

uCoarseRestrNodal = zeros( length(xCoarse) );
for (i, el) in enumerate( coarseMesh.mElements )
    basisFunVal = aggmg.evaluate_local_modal_basis_fun( coarseMesh.mP, el.mBoundingBox, 
        xCoarse[ (2*i-1):2*i ] );
    uCoarseRestrNodal[ 2*i-1 ] = la.dot( uCoarseRestr[ el.mNodesInd ], basisFunVal[1,:] );
    uCoarseRestrNodal[ 2*i ] = la.dot( uCoarseRestr[ el.mNodesInd ], basisFunVal[2,:] );
end

# calculate errors in solution
l2ErrorReg = 0.0;
l2ErrorRestr = 0.0;

for coarseEl in coarseMesh.mElements
    count = 0;
    for subElInd in coarseEl.mSubAggElementInds
        fineEl = fineMesh.mElements[ subElInd ];
        for (k, baseElInd) in enumerate( fineEl.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            for (l, xGQ) in enumerate( fineMesh.mGaussQuadNodes )
                global l2ErrorReg += baseEl.mJacobian * fineMesh.mGaussQuadWeights[l] * 
                    ( la.dot( uCoarse[coarseEl.mNodesInd], coarseEl.mBasisGQFunVal[count+k][l,:] ) - 
                    la.dot( uFine[fineEl.mNodesInd], fineEl.mBasisGQFunVal[k][l,:] ) )^2;
                global l2ErrorRestr += baseEl.mJacobian * fineMesh.mGaussQuadWeights[l] * 
                    ( la.dot( uCoarseRestr[coarseEl.mNodesInd], coarseEl.mBasisGQFunVal[count+k][l,:] ) - 
                    la.dot( uFine[fineEl.mNodesInd], fineEl.mBasisGQFunVal[k][l,:] ) )^2;
            end
        end
        count += length( fineEl.mBaseElementInds );
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorRestr = sqrt(l2ErrorRestr);

println("\nDifferences in solutions when restricting:")
println("||uCoarse - uFine||: ", l2ErrorReg)
println("||uCoarseRestr - uFine||: ", l2ErrorRestr)

# plot
s1 = ml.get_default_msession();

ml.put_variable(s1, :xCoarse, xCoarse[permCoarse]);
ml.put_variable(s1, :uCoarse, uCoarseNodal[permCoarse]);
ml.put_variable(s1, :uCoarseRestr, uCoarseRestrNodal[permCoarse]);
ml.put_variable(s1, :xFine, xFine[permFine]);
ml.put_variable(s1, :uFine, uFineNodal[permFine]);


ml.eval_string(s1, "figure();");
ml.eval_string(s1, "hold on;");
ml.eval_string(s1, "plot(xCoarse, uCoarse, '-s', 'LineWidth', 4, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xCoarse, uCoarseRestr, '-*', 'LineWidth', 2, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xFine, uFine, '-o', 'LineWidth', 2, 'MarkerSize', 10)");

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

# agglomerate twice
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
fineMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

agg2 = Vector{Vector{Int64}}( undef, div(n, 4) );
for i in 1:length(agg2)
    agg2[i] = (2*i-1):(2*i);
end
coarseMesh = aggmg.AgglomeratedDgMeshN( pAgg, agg2, fineMesh, baseMesh );

L = aggmg.aggdg_aggdg_interpolation( coarseMesh, fineMesh, baseMesh );

# get coarse solution vector
xCoarse = zeros( 2 * length(coarseMesh.mElements) );
uCoarseNodal = zeros( length(xCoarse) );
for (i, el) in enumerate( coarseMesh.mElements )
    xCoarse[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if coarseMesh.mP == 0
        uCoarseNodal[ (2*i-1):2*i ] .= 
            ( u_exact( xCoarse[2*i-1] ) + u_exact( xCoarse[2*i] ) ) / 2.0;
    elseif coarseMesh.mP == 1
        uCoarseNodal[ (2*i-1):2*i ] = u_exact.( xCoarse[ (2*i-1):2*i ] );
    end
end
permCoarse = sortperm( xCoarse );

uCoarse = zeros( coarseMesh.mNumNodes );
for (i, el) in enumerate( coarseMesh.mElements )
    if coarseMesh.mP == 0
        uCoarse[el.mNodesInd] .= ( uCoarseNodal[2*i-1] + uCoarseNodal[2*i] ) / 2.0;
    elseif coarseMesh.mP == 1
        uCoarse[el.mNodesInd[1]] = ( uCoarseNodal[2*i-1] + uCoarseNodal[2*i] ) / 2.0;
        uCoarse[el.mNodesInd[2]] = ( uCoarseNodal[2*i] - uCoarseNodal[2*i-1] ) / 2.0;
    end
end

# get fine solution vector
xFine = zeros( 2 * length(fineMesh.mElements) );
uFineNodal = zeros( length(xFine) );
for (i, el) in enumerate( fineMesh.mElements )
    xFine[ (2*i-1):2*i ] = el.mBoundingBox + [1e-14, -1e-14];
    if fineMesh.mP == 0
        uFineNodal[ (2*i-1):2*i ] .= 
            ( u_exact( xFine[2*i-1] ) + u_exact( xFine[2*i] ) ) / 2.0;
    elseif fineMesh.mP == 1
        uFineNodal[ (2*i-1):2*i ] = u_exact.( xFine[ (2*i-1):2*i ] );
    end
end
permFine = sortperm( xFine );

uFine = zeros( fineMesh.mNumNodes );
for (i, el) in enumerate( fineMesh.mElements )
    if fineMesh.mP == 0
        uFine[el.mNodesInd] .= ( uFineNodal[2*i-1] + uFineNodal[2*i] ) / 2.0;
    elseif fineMesh.mP == 1
        uFine[el.mNodesInd[1]] = ( uFineNodal[2*i-1] + uFineNodal[2*i] ) / 2.0;
        uFine[el.mNodesInd[2]] = ( uFineNodal[2*i] - uFineNodal[2*i-1] ) / 2.0;
    end
end

# find interpolated solution, and nodal form of it
uFineInterp = L * uCoarse;

uFineInterpNodal = zeros( length(xFine) );
for (i, el) in enumerate( fineMesh.mElements )
    basisFunVal = aggmg.evaluate_local_modal_basis_fun( fineMesh.mP, el.mBoundingBox, 
        xFine[ (2*i-1):2*i ] );
    uFineInterpNodal[ 2*i-1 ] = la.dot( uFineInterp[ el.mNodesInd ], basisFunVal[1,:] );
    uFineInterpNodal[ 2*i ] = la.dot( uFineInterp[ el.mNodesInd ], basisFunVal[2,:] );
end

# calculate errors in solution
l2ErrorReg = 0.0;
l2ErrorInterp = 0.0;

for coarseEl in coarseMesh.mElements
    count = 0;
    for subElInd in coarseEl.mSubAggElementInds
        fineEl = fineMesh.mElements[ subElInd ];
        for (k, baseElInd) in enumerate( fineEl.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            for (l, xGQ) in enumerate( fineMesh.mGaussQuadNodes )
                global l2ErrorReg += baseEl.mJacobian * fineMesh.mGaussQuadWeights[l] * 
                    ( la.dot( uCoarse[coarseEl.mNodesInd], coarseEl.mBasisGQFunVal[count+k][l,:] ) - 
                    la.dot( uFine[fineEl.mNodesInd], fineEl.mBasisGQFunVal[k][l,:] ) )^2;
                global l2ErrorInterp += baseEl.mJacobian * fineMesh.mGaussQuadWeights[l] * 
                    ( la.dot( uCoarse[coarseEl.mNodesInd], coarseEl.mBasisGQFunVal[count+k][l,:] ) - 
                    la.dot( uFineInterp[fineEl.mNodesInd], fineEl.mBasisGQFunVal[k][l,:] ) )^2;
            end
        end
        count += length( fineEl.mBaseElementInds );
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorInterp = sqrt(l2ErrorInterp);

println("\nDifferences in discontinuous solutions when interpolating:")
println("||uCoarse - uFine||: ", l2ErrorReg)
println("||uCoarse - uFineInterp||: ", l2ErrorInterp)

# plot
s1 = ml.get_default_msession();

ml.put_variable(s1, :xCoarse, xCoarse[permCoarse]);
ml.put_variable(s1, :uCoarse, uCoarseNodal[permCoarse]);
ml.put_variable(s1, :xFine, xFine[permFine]);
ml.put_variable(s1, :uFine, uFineNodal[permFine]);
ml.put_variable(s1, :uFineInterp, uFineInterpNodal[permFine]);

ml.eval_string(s1, "figure();");
ml.eval_string(s1, "hold on;");
ml.eval_string(s1, "plot(xCoarse, uCoarse, '-s', 'LineWidth', 4, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xFine, uFine, '-o', 'LineWidth', 2, 'MarkerSize', 10)");
ml.eval_string(s1, "plot(xFine, uFineInterp, '-*', 'LineWidth', 2, 'MarkerSize', 10)");