include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la
import SparseArrays as sp

# basic parameters
xin = 0.0;
xout = 1.0;

n = 16;
CDir = 1000.0*n;

p = 2;

############################################################################################
# test number of iterations to convergence on solve for different smoothers
############################################################################################

func(x) = 1.0;
u_exact(x) = -1/2*x^2 + x;

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:dir, u_exact(xin)), (:dir, u_exact(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

dgMesh = aggmg.DgMesh( mesh, p );

# mesh operators and rhs
G, D, C = aggmg.dg_flux_operators( dgMesh, mesh, bdCond, CDir );
A = C - D * ( dgMesh.mMassMatrixLU \ G );

f, r = aggmg.dg_flux_rhs( dgMesh, mesh, func, bdCond, CDir );
b = f - D * ( dgMesh.mMassMatrixLU \ r );

# smoother on the mesh
smoother1 = aggmg.dg_smoother( dgMesh, A, :blockJac );
smoother2 = aggmg.dg_smoother( dgMesh, A, :jac );

u0 = zeros(size(A,2));
r0 = b - A*u0;
u1 = u0 + aggmg.apply_smoother(smoother1, r0);
r1 = b - A*u1;

u1, iter1, res1, err1 = aggmg.iterative_smoother_solve( A, smoother1, u0, b; 
    maxiter = 10^4, alpha = 2.0/3.0 );
u2, iter2, res2, err2 = aggmg.iterative_smoother_solve( A, smoother2, u0, b; 
    maxiter = 10^4, alpha = 2.0/3.0 );

############################################################################################
# test that block jacobi smoother smooths out sine waves
############################################################################################

func(x) = -pi^2*sin(pi*x);
u_exact(x) = sin(pi*x);

mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:dir, u_exact(xin)), (:dir, u_exact(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

dgMesh = aggmg.DgMesh( mesh, p );

G, D, C = aggmg.dg_flux_operators( dgMesh, mesh, bdCond, CDir );
A = C - D * ( dgMesh.mMassMatrixLU \ G );

f, r = aggmg.dg_flux_rhs( dgMesh, mesh, func, bdCond, CDir );
b = f - D * ( dgMesh.mMassMatrixLU \ r );

smoother1 = aggmg.dg_smoother( dgMesh, A, :blockJac );
smoother2 = aggmg.dg_smoother( dgMesh, A, :jac );

s1 = ml.get_default_msession();

x = zeros( dgMesh.mNumNodes );
for el in dgMesh.mElements
    x[ el.mNodesInd ] = el.mNodesX;
end
perm = sortperm(x);

for i in 1:10
    u_fun(x) = sin(i*pi*x);
    u = u_fun.(x);

    uBlockJac = u;
    uJac = u;
    for j in 1:10
        uBlockJac = uBlockJac - aggmg.apply_smoother( smoother1, A*uBlockJac; 
            alpha = 2.0/3.0 );
        uJac = uJac - aggmg.apply_smoother( smoother2, A*uJac; alpha = 2.0/3.0 );
    end

    ml.put_variable(s1, :x, x[perm]);
    ml.put_variable(s1, :u, u[perm]);
    ml.put_variable(s1, :uBlockJac, uBlockJac[perm]);
    ml.put_variable(s1, :uJac, uJac[perm]);
    ml.put_variable(s1, :i, i);

    ml.eval_string(s1, "figure(cast(i, 'like', 1));");
    ml.eval_string(s1, "plot(x, u, '-', 'LineWidth', 3); hold on;");
    ml.eval_string(s1, "plot(x, uBlockJac, '-o', 'LineWidth', 2, 'MarkerSize', 10);");
    ml.eval_string(s1, "plot(x, uJac, '-s', 'LineWidth', 2, 'MarkerSize', 10);");
    ml.eval_string(s1, "legend('Original', 'Block Jacobi', 'Jacobi'); hold off;")
end

R1 = Matrix( la.I(size(A,1)) - aggmg.apply_smoother(smoother1, A; alpha = 2.0/3.0) );
R2 = Matrix( la.I(size(A,1)) - aggmg.apply_smoother(smoother2, A; alpha = 2.0/3.0) );

S1, V1 = la.eigen(R1);
S2, V2 = la.eigen(R2);

ml.eval_string(s1, "figure(11);");
ml.put_variable(s1, :S1, real(S1));
ml.put_variable(s1, :S2, real(S2));
ml.eval_string(s1, "plot(S1, '-o', 'LineWidth', 3); hold on;");
ml.eval_string(s1, "plot(S2, '-s', 'LineWidth', 3); hold on;");
ml.eval_string(s1, "legend('BlockJacobi', 'Jacobi'); hold off;")

ml.eval_string(s1, "figure(12);");
for i in size(V1,2)-3:size(V1,2)
    ml.put_variable(s1, :v, real(V1[:,i][perm]));
    ml.eval_string(s1, "plot(x, v, '-', 'LineWidth', 3); hold on;");
end
ml.eval_string(s1, "hold off;")

ml.eval_string(s1, "figure(12);");
for i in size(V2,2)-3:size(V2,2)
    ml.put_variable(s1, :v, real(V2[:,i][perm]));
    ml.eval_string(s1, "plot(x, v, '-', 'LineWidth', 3); hold on;");
end
ml.eval_string(s1, "hold off;")