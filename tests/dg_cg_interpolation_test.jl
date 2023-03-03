include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;

highP = 1;
lowP = 1;

n = 16;
CDir = 1.0*n;
interpFlag = 1;

############################################################################################
# test operators on small and large mesh
############################################################################################

func(x) = cos(x);
u_exact(x) = cos(x);

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, -sin(xin)), (:dir, cos(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

highMesh = aggmg.CgMesh( mesh, highP );
lowMesh = aggmg.DgMesh( mesh, lowP );

L0 = aggmg.dg_cg_interpolation( lowMesh, highMesh, mesh, 0 );
for i in eachindex(L0)
    if abs(L0[i]) <= 1e-14
        L0[i] = 0.0;
    end
end
L0 = sp.sparse(L0);
L1 = aggmg.dg_cg_interpolation( lowMesh, highMesh, mesh, 1 );
L2 = aggmg.dg_cg_interpolation( lowMesh, highMesh, mesh, 2 );

lowG, lowD, lowC = aggmg.dg_flux_operators( lowMesh, mesh, bdCond, CDir );
lowA = lowC - lowD * ( lowMesh.mMassMatrixLU \ lowG );

highA = aggmg.cg_stiffness( highMesh, bdCond );

############################################################################################
# test interpolation of solutions themselves
############################################################################################

u_exact(x) = -x.^3 + 2*x + 3.0;
ux_exact(x) = -3*x.^2 + 2.0;
# u_exact(x) = 2.0;
# ux_exact(x) = 0.0;

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ (:neu, ux_exact(xin)), (:dir, u_exact(xout)) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

highMesh = aggmg.CgMesh( mesh, highP );
lowMesh = aggmg.DgMesh( mesh, lowP );

L = aggmg.dg_cg_interpolation( lowMesh, highMesh, mesh, interpFlag );

uLow = zeros( lowMesh.mNumNodes );
for el in lowMesh.mElements
    for (i, x) in enumerate( el.mNodesX )
        uLow[ el.mNodesInd[i] ] = u_exact( x );
    end
end

uHigh = zeros( highMesh.mNumNodes );
for el in highMesh.mElements
    for (i, x) in enumerate( el.mNodesX )
        uHigh[ el.mNodesInd[i] ] = u_exact( x );
    end
end
uHighInterp = L*uLow;

l2ErrorReg = 0.0;
l2ErrorInterp = 0.0;

gaussQuadNodes, gaussQuadWeights = aggmg.gauss_quad( 2*highP );
lowBasisGQFunVal = aggmg.evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
    gaussQuadNodes);
highBasisGQFunVal = aggmg.evaluate_nodal_basis_fun( highMesh.mRefEl.mBasisFunCoeff, 
    gaussQuadNodes);
for (i, lowEl) in enumerate(lowMesh.mElements)
    highEl = highMesh.mElements[i];
    for (l, xGQ) in enumerate(gaussQuadNodes)
        global l2ErrorReg += lowEl.mJacobian * gaussQuadWeights[l] * 
            ( la.dot( uHigh[highEl.mNodesInd], highBasisGQFunVal[l,:] )  - 
            la.dot( uLow[lowEl.mNodesInd], lowBasisGQFunVal[l,:] ) )^2;
        global l2ErrorInterp += lowEl.mJacobian * gaussQuadWeights[l] * 
            ( la.dot( uHighInterp[highEl.mNodesInd], highBasisGQFunVal[l,:] )  - 
            la.dot( uLow[lowEl.mNodesInd], lowBasisGQFunVal[l,:] ) )^2;
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorInterp = sqrt(l2ErrorInterp);

println("\nDifferences in solutions when interpolating:")
println("||uLow - uHigh||: ", l2ErrorReg)
println("||uLow - uHighInterp||: ", l2ErrorInterp)

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

highMesh = aggmg.CgMesh( mesh, highP );
lowMesh = aggmg.DgMesh( mesh, lowP );

L = aggmg.dg_cg_interpolation( lowMesh, highMesh, mesh, interpFlag );

uLow = zeros( lowMesh.mNumNodes );
for el in lowMesh.mElements
    for (i, x) in enumerate( el.mNodesX )
        uLow[ el.mNodesInd[i] ] = u_exact( x );
    end
end

uHigh = zeros( highMesh.mNumNodes );
for el in highMesh.mElements
    for (i, x) in enumerate( el.mNodesX )
        uHigh[ el.mNodesInd[i] ] = u_exact( x );
    end
end

uLowRestr = lowMesh.mMassMatrixLU \ ( L' * ( highMesh.mMassMatrix * uHigh ) );

l2ErrorReg = 0.0;
l2ErrorRestr = 0.0;

gaussQuadNodes, gaussQuadWeights = aggmg.gauss_quad( 2*highP );
lowBasisGQFunVal = aggmg.evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
    gaussQuadNodes);
highBasisGQFunVal = aggmg.evaluate_nodal_basis_fun( highMesh.mRefEl.mBasisFunCoeff, 
    gaussQuadNodes);
for (i, lowEl) in enumerate(lowMesh.mElements)
    highEl = highMesh.mElements[i];
    for (l, xGQ) in enumerate(gaussQuadNodes)
        global l2ErrorReg += lowEl.mJacobian * gaussQuadWeights[l] * 
            ( la.dot( uHigh[highEl.mNodesInd], highBasisGQFunVal[l,:] )  - 
            la.dot( uLow[lowEl.mNodesInd], lowBasisGQFunVal[l,:] ) )^2;
        global l2ErrorRestr += lowEl.mJacobian * gaussQuadWeights[l] * 
            ( la.dot( uHigh[highEl.mNodesInd], highBasisGQFunVal[l,:] )  - 
            la.dot( uLowRestr[lowEl.mNodesInd], lowBasisGQFunVal[l,:] ) )^2;
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorInterp = sqrt(l2ErrorRestr);

println("\nDifferences in solutions when restricting:")
println("||uLow - uHigh||: ", l2ErrorReg)
println("||uLowRestr - uHigh||: ", l2ErrorRestr)

############################################################################################
# test interpolation of discontinuous solution onto CG mesh 
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

highMesh = aggmg.CgMesh( mesh, highP );
lowMesh = aggmg.DgMesh( mesh, lowP );

L = aggmg.dg_cg_interpolation( lowMesh, highMesh, mesh, interpFlag );

uLow = zeros( lowMesh.mNumNodes );
for el in lowMesh.mElements
    for (i, x) in enumerate( el.mNodesX )
        if i == 1
            uLow[ el.mNodesInd[i] ] = u_exact( x + 1e-14 );
        elseif i == 2
            uLow[ el.mNodesInd[i] ] = u_exact( x - 1e-14 );
        else
            uLow[ el.mNodesInd[i] ] = u_exact( x );
        end
    end
end

uHigh = zeros( highMesh.mNumNodes );
for el in highMesh.mElements
    for (i, x) in enumerate( el.mNodesX )
        if i == 1
            uHigh[ el.mNodesInd[i] ] = u_exact( x + 1e-14 );
        elseif i == 2
            uHigh[ el.mNodesInd[i] ] = u_exact( x - 1e-14 );
        else
            uHigh[ el.mNodesInd[i] ] = u_exact( x );
        end
    end
end
uHighInterp = L*uLow;

l2ErrorReg = 0.0;
l2ErrorInterp = 0.0;

gaussQuadNodes, gaussQuadWeights = aggmg.gauss_quad( 2*highP );
lowBasisGQFunVal = aggmg.evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
    gaussQuadNodes);
highBasisGQFunVal = aggmg.evaluate_nodal_basis_fun( highMesh.mRefEl.mBasisFunCoeff, 
    gaussQuadNodes);
for (i, lowEl) in enumerate(lowMesh.mElements)
    highEl = highMesh.mElements[i];
    for (l, xGQ) in enumerate(gaussQuadNodes)
        global l2ErrorReg += lowEl.mJacobian * gaussQuadWeights[l] * 
            ( la.dot( uHigh[highEl.mNodesInd], highBasisGQFunVal[l,:] )  - 
            la.dot( uLow[lowEl.mNodesInd], lowBasisGQFunVal[l,:] ) )^2;
        global l2ErrorInterp += lowEl.mJacobian * gaussQuadWeights[l] * 
            ( la.dot( uHighInterp[highEl.mNodesInd], highBasisGQFunVal[l,:] )  - 
            la.dot( uLow[lowEl.mNodesInd], lowBasisGQFunVal[l,:] ) )^2;
    end
end
l2ErrorReg = sqrt(l2ErrorReg);
l2ErrorInterp = sqrt(l2ErrorInterp);

println("\nDifferences in discontinuous solutions when interpolating:")
println("||uLow - uHigh||: ", l2ErrorReg)
println("||uLow - uHighInterp||: ", l2ErrorInterp)

# plot the solutions to see what is up
xHigh = zeros( highMesh.mNumNodes );
for el in highMesh.mElements
    xHigh[ el.mNodesInd ] = el.mNodesX;
end
permHigh = sortperm(xHigh);

xLow = zeros( lowMesh.mNumNodes );
for el in lowMesh.mElements
    xLow[ el.mNodesInd ] = el.mNodesX;
end
permLow = sortperm(xLow);

s1 = ml.get_default_msession();
ml.put_variable(s1, :xLow, xLow[permLow]);
ml.put_variable(s1, :xHigh, xHigh[permHigh]);
ml.put_variable(s1, :uLow, uLow[permLow]);
ml.put_variable(s1, :uHigh, uHigh[permHigh]);
ml.put_variable(s1, :uHighInterp, uHighInterp[permHigh]);

ml.eval_string(s1, "figure();");
ml.eval_string(s1, "plot(xLow, uLow, '-o', 'LineWidth', 2, 'MarkerSize', 10); hold on;");
ml.eval_string(s1, "plot(xHigh, uHigh, '-s', 'LineWidth', 2, 'MarkerSize', 10);");
ml.eval_string(s1, "plot(xHigh, uHighInterp, '-*', 'LineWidth', 2, 'MarkerSize', 10);");
ml.eval_string(s1, "legend('uLow', 'uHigh', 'uHighInterp'); hold off;")

xTemp = collect(range(-1, 1, 21));
lowBasisFunVal = aggmg.evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, xTemp );
highBasisFunVal = aggmg.evaluate_nodal_basis_fun( highMesh.mRefEl.mBasisFunCoeff, xTemp );

xNew = zeros( length(lowMesh.mElements) * length(xTemp) );
uLowNew = zeros( length(xNew) );
uHighNew = zeros( length(xNew) );
uHighInterpNew = zeros( length(xNew) );

for (i, lowEl) in enumerate(lowMesh.mElements)
    highEl = highMesh.mElements[i];
    x1 = highEl.mNodesX[1];
    x2 = highEl.mNodesX[2];
    inds = (i-1)*length(xTemp)+1:i*length(xTemp);

    xNew[ inds ] = (x1 + x2)/2.0 .+ (x2-x1)/2.0*xTemp;
    uLowNew[ inds ] = lowBasisFunVal * uLow[lowEl.mNodesInd];
    uHighNew[ inds ] = highBasisFunVal * uHigh[highEl.mNodesInd];
    uHighInterpNew[ inds ] = highBasisFunVal * uHighInterp[highEl.mNodesInd];
end

ml.put_variable(s1, :xNew, xNew);
ml.put_variable(s1, :uLowNew, uLowNew);
ml.put_variable(s1, :uHighNew, uHighNew);
ml.put_variable(s1, :uHighInterpNew, uHighInterpNew);

ml.eval_string(s1, "figure();");
ml.eval_string(s1, "plot(xNew, uLowNew, '-o', 'LineWidth', 2, 'MarkerSize', 10); hold on;");
ml.eval_string(s1, "plot(xNew, uHighNew, '-s', 'LineWidth', 2, 'MarkerSize', 10);");
ml.eval_string(s1, "plot(xNew, uHighInterpNew, '-*', 'LineWidth', 2, 'MarkerSize', 10);");
ml.eval_string(s1, "legend('uLow', 'uHigh', 'uHighInterp'); hold off;")