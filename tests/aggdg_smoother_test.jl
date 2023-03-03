include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;

p = 1;
pAgg = 1;

n = 32;
CDir = 1000.0*n;

############################################################################################
# test number of iterations to convergence on solve for different smoothers
############################################################################################

func(x) = 1.0;
u_exact(x) = -1/2*x^2 + x;

# create basic mesh
mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ ( :dir, u_exact(xin) ), ( :dir, u_exact(xout) ) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

# create base mesh
baseMesh = aggmg.DgMesh( mesh, p );

# agglomerate once
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
aggMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

# mesh operators and rhs
G, D, C = aggmg.dg_flux_operators( aggMesh, baseMesh, bdCond, CDir );
f, r = aggmg.dg_flux_rhs( aggMesh, baseMesh, func, bdCond, CDir );

b = f - D * ( aggMesh.mMassMatrixLU \ r );
A = C - D * ( aggMesh.mMassMatrixLU \ G );

# smoother on mesh
smoother1 = aggmg.dg_smoother( aggMesh, A, :blockJac );
smoother2 = aggmg.dg_smoother( aggMesh, A, :jac );

# smooth once
u0 = zeros( size(A,2) );
r0 = b - A*u0;
u1 = u0 + aggmg.apply_smoother( smoother1, r0 );
r1 = b - A*u1;

# do smoother solves 
u1, iter1, res1, err1 = aggmg.iterative_smoother_solve( A, smoother1, u0, b; 
    maxiter = 10^4, alpha = 2.0/3.0 );
u2, iter2, res2, err2 = aggmg.iterative_smoother_solve( A, smoother2, u0, b; 
    maxiter = 10^4, alpha = 2.0/3.0 );

############################################################################################
# test that block jacobi smoother smooths out sine waves
############################################################################################

func(x) = -pi^2*sin(pi*x);
u_exact(x) = sin(pi*x);
xin = 0.0;
xout = 1.0;

# create basic mesh
mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [ ( :dir, u_exact(xin) ), ( :dir, u_exact(xout) ) ];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

# create base mesh
baseMesh = aggmg.DgMesh( mesh, p );

# agglomerate once
agg = Vector{Vector{Int64}}( undef, div(n, 2) );
for i in 1:length(agg)
    agg[i] = (2*i-1):(2*i);
end
aggMesh = aggmg.AgglomeratedDgMesh1( pAgg, agg, mesh, baseMesh );

# mesh operators and rhs
G, D, C = aggmg.dg_flux_operators( aggMesh, baseMesh, bdCond, CDir );
f, r = aggmg.dg_flux_rhs( aggMesh, baseMesh, func, bdCond, CDir );

b = f - D * ( aggMesh.mMassMatrixLU \ r );
A = C - D * ( aggMesh.mMassMatrixLU \ G );

# smoother on mesh
smoother1 = aggmg.dg_smoother( aggMesh, A, :blockJac );
smoother2 = aggmg.dg_smoother( aggMesh, A, :jac );

# create vector of xvalues of element vertices
x = zeros( 2 * length(aggMesh.mElements) );
for (i, el) in enumerate( aggMesh.mElements )
    x[ (2*i-1):2*i ] = el.mBoundingBox;
end
perm = sortperm(x);

s1 = ml.get_default_msession();

for i in 1:10
    u_fun(x) = sin(i*pi*x);
    uNodal = u_fun.(x);

    # convert to modal
    uModal = zeros( aggMesh.mNumNodes );
    for (j, el) in enumerate( aggMesh.mElements )
        if aggMesh.mP == 0
            uModal[el.mNodesInd] .= ( uNodal[2*j-1] + uNodal[2*j] ) / 2.0;
        elseif aggMesh.mP == 1
            uModal[el.mNodesInd[1]] = ( uNodal[2*j-1] + uNodal[2*j] ) / 2.0;
            uModal[el.mNodesInd[2]] = ( uNodal[2*j] - uNodal[2*j-1] ) / 2.0;
        end
    end

    # do smoother solves
    uBlockJac = uModal;
    uJac = uModal;
    for j in 1:5
        uBlockJac = uBlockJac - aggmg.apply_smoother( smoother1, A*uBlockJac; 
            alpha = 2.0/3.0 );
        uJac = uJac - aggmg.apply_smoother( smoother2, A*uJac; alpha = 2.0/3.0 );
    end

    # get back into some nodal form
    uBlockJacPlot = zeros( length(x) );
    uJacPlot = zeros( length(x) );
    for (j, el) in enumerate(aggMesh.mElements)
        basisFunVal = aggmg.evaluate_local_modal_basis_fun( aggMesh.mP, el.mBoundingBox, 
            x[ (2*j-1):2*j ] );

        uBlockJacPlot[ 2*j-1 ] = la.dot( uBlockJac[ el.mNodesInd ], basisFunVal[1,:] );
        uBlockJacPlot[ 2*j ] = la.dot( uBlockJac[ el.mNodesInd ], basisFunVal[2,:] );

        uJacPlot[ 2*j-1 ] = la.dot( uJac[ el.mNodesInd ], basisFunVal[1,:] );
        uJacPlot[ 2*j ] = la.dot( uJac[ el.mNodesInd ], basisFunVal[2,:] );
    end

    # plot
    ml.put_variable(s1, :x, x[perm]);
    ml.put_variable(s1, :u, uNodal[perm]);
    ml.put_variable(s1, :uBlockJac, uBlockJacPlot[perm]);
    ml.put_variable(s1, :uJac, uJacPlot[perm]);
    ml.put_variable(s1, :i, i);

    ml.eval_string(s1, "figure(cast(i, 'like', 1));");
    ml.eval_string(s1, "plot(x, u, '-', 'LineWidth', 3); hold on;");
    ml.eval_string(s1, "plot(x, uBlockJac, '-o', 'LineWidth', 2, 'MarkerSize', 10);");
    ml.eval_string(s1, "plot(x, uJac, '-s', 'LineWidth', 2, 'MarkerSize', 10);");
    ml.eval_string(s1, "legend('Original', 'Block Jacobi', 'Jacobi'); hold off;")
end

# get eigenvalues and eigenvectors of smoother matrices
R1 = Matrix( la.I( size(A,1) ) - aggmg.apply_smoother( smoother1, A; alpha = 2.0/3.0 ) );
R2 = Matrix( la.I( size(A,1) ) - aggmg.apply_smoother( smoother2, A; alpha = 2.0/3.0 ) );

S1, V1 = la.eigen(R1);
S2, V2 = la.eigen(R2);

# plot eigenvalues
ml.eval_string(s1, "figure(11);");
ml.put_variable(s1, :S1, real(S1));
ml.put_variable(s1, :S2, real(S2));
ml.eval_string(s1, "plot(S1, '-o', 'LineWidth', 3); hold on;");
ml.eval_string(s1, "plot(S2, '-s', 'LineWidth', 3); hold on;");
ml.eval_string(s1, "legend('BlockJacobi', 'Jacobi'); hold off;")

# ml.eval_string(s1, "figure(12);");
# for i in size(V1,2)-3:size(V1,2)
#     ml.put_variable(s1, :v, real(V1[:,i]));
#     ml.eval_string(s1, "plot(v, '-', 'LineWidth', 3); hold on;");
# end
# ml.eval_string(s1, "hold off;")

# ml.eval_string(s1, "figure(13);");
# for i in size(V2,2)-3:size(V2,2)
#     ml.put_variable(s1, :v, real(V2[:,i]));
#     ml.eval_string(s1, "plot(v, '-', 'LineWidth', 3); hold on;");
# end
# ml.eval_string(s1, "hold off;")