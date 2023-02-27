include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;

highP = 4;
lowP = 2;

n = 3;
CDir = 1.0*n; 

############################################################################################
# test operators on small and large mesh
############################################################################################

func(x) = cos(x);
u_exact(x) = cos(x);

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, -sin(xin)), (:dir, cos(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

highMesh = aggmg.DgMesh( mesh, highP );
lowMesh = aggmg.DgMesh( mesh, lowP );

L = aggmg.dg_dg_interpolation( lowMesh, highMesh );
L2 = aggmg.dg_dg_interpolation2( lowMesh, highMesh );

lowG, lowD, lowC = aggmg.dg_flux_operators( lowMesh, mesh, bdCond, CDir );
lowA = lowC - lowD * sp.sparse(Matrix(lowMesh.mMassMatrix) \ lowG);

highG, highD, highC = aggmg.dg_flux_operators( highMesh, mesh, bdCond, CDir );
highA = highC - highD * sp.sparse(Matrix(highMesh.mMassMatrix) \ highG);

println("||lowG - L'*highG*L||: ", la.norm(lowG - L'*highG*L));
println("||lowD - L'*highD*L||: ", la.norm(lowD - L'*highD*L));
println("||lowC - L'*highC*L||: ", la.norm(lowC - L'*highC*L));
println("||lowM - L'*highM*L||: ", la.norm(lowMesh.mMassMatrix - L'*highMesh.mMassMatrix*L));

############################################################################################
# test interpolation of solutions themselves
############################################################################################

u_exact(x) = -x.^4 + 2*x + 3.0
ux_exact(x) = -4*x.^3 + 2.0

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, ux_exact(0.0)), (:dir, u_exact(1.0))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

highMesh = aggmg.DgMesh( mesh, highP );
lowMesh = aggmg.DgMesh( mesh, lowP );

L = aggmg.dg_dg_interpolation( lowMesh, highMesh );

uLow = zeros( lowMesh.mNumNodes );
for el in lowMesh.mElements
    for (i, x) in enumerate( el.mNodesX );
        uLow[ el.mNodesInd[i] ] = u_exact( x );
    end
end

uHigh = zeros( highMesh.mNumNodes );
for el in highMesh.mElements
    for (i, x) in enumerate( el.mNodesX );
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

u_exact(x) = -x.^2 + 2*x + 3.0
ux_exact(x) = -2*x + 2.0

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, ux_exact(0.0)), (:dir, u_exact(1.0))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

highMesh = aggmg.DgMesh( mesh, highP );
lowMesh = aggmg.DgMesh( mesh, lowP );

L = aggmg.dg_dg_interpolation( lowMesh, highMesh );

uLow = zeros( lowMesh.mNumNodes );
for el in lowMesh.mElements
    for (i, x) in enumerate( el.mNodesX );
        uLow[ el.mNodesInd[i] ] = u_exact( x );
    end
end

uHigh = zeros( highMesh.mNumNodes );
for el in highMesh.mElements
    for (i, x) in enumerate( el.mNodesX );
        uHigh[ el.mNodesInd[i] ] = u_exact( x );
    end
end

uLowRestr = lowMesh.mMassMatrixLU \ (L' * ( highMesh.mMassMatrix * uHigh ))

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