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

############################################################################################
# test operators on small and large mesh
############################################################################################

func(x) = cos(x);
u_exact(x) = cos(x);


mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, -sin(xin)), (:dir, cos(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

highMesh = aggmg.CgMesh( mesh, highP );
lowMesh = aggmg.CgMesh( mesh, lowP );

L = aggmg.cg_cg_interpolation( lowMesh, highMesh );
# L2 = aggmg.cg_cg_interpolation2(lowMesh, highMesh);
# for i in eachindex(L2)
#     if abs(L2[i]) <= 1e-14
#         L2[i] = 0.0;
#     end
# end
# L2 = sp.sparse(L2);

lowA = aggmg.cg_stiffness( lowMesh, bdCond );
highA = aggmg.cg_stiffness( highMesh, bdCond );

println("||lowA - L'*highA*L||: ", la.norm(lowA - L'*highA*L));
display(sp.sparse(lowA - L'*highA*L));

############################################################################################
# test interpolation of solutions themselves
############################################################################################

u_exact(x) = -x.^4 + 2*x + 3.0
ux_exact(x) = -4*x.^3 + 2.0

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:neu, ux_exact(0.0)), (:dir, u_exact(1.0))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

highMesh = aggmg.CgMesh( mesh, highP );
lowMesh = aggmg.CgMesh( mesh, lowP );

L = aggmg.cg_cg_interpolation( lowMesh, highMesh );

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

highMesh = aggmg.CgMesh( mesh, highP );
lowMesh = aggmg.CgMesh( mesh, lowP );

L = aggmg.cg_cg_interpolation( lowMesh, highMesh );

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