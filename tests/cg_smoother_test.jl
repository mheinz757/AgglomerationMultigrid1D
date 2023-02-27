include("mesh_generator.jl")

import MATLAB as ml
import LinearAlgebra as la

# basic parameters
xin = 0.0;
xout = 1.0;

n = 16;

p = 4;

############################################################################################
# test number of iterations to convergence on solve for different smoothers
############################################################################################

func(x) = 1.0;
u_exact(x) = -1/2*x^2 + x;

# create basic mesh
mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:dir, u_exact(xin)), (:dir, u_exact(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

# create cgMesh
cgMesh = aggmg.CgMesh( mesh, p );

# mesh operators and rhs
A, b = aggmg.cg_stiffness_and_rhs( cgMesh, mesh, func, bdCond );

# smoothers on mesh
smoother1 = aggmg.cg_smoother( cgMesh, A, :jac );
smoother2 = aggmg.cg_smoother( cgMesh, A, :addSchwarz );
smoother3 = aggmg.cg_smoother( cgMesh, A, :hybridSchwarz );

# smooth once
u0 = zeros(size(A,2));
r0 = b - A*u0;
u1 = u0 + aggmg.apply_smoother(smoother3, r0);
r1 = b - A*u1;

# do smoother solves
u1, iter1, res1, err1 = aggmg.iterative_smoother_solve(A, smoother1, u0, b; 
    maxiter = 10^4, alpha = 0.5);
u2, iter2, res2, err2 = aggmg.iterative_smoother_solve(A, smoother2, u0, b; 
    maxiter = 10^4, alpha = 0.5);
u3, iter3, res3, err3 = aggmg.iterative_smoother_solve(A, smoother3, u0, b; 
    maxiter = 10^4);

############################################################################################
# test that jacobi smoother smooths out sine waves
############################################################################################

func(x) = -pi^2*sin(pi*x);
u_exact(x) = sin(pi*x);

# create basic mesh
mesh = create_uniform_mesh( n, xin, xout );
mBdCond = [(:dir, u_exact(xin)), (:dir, u_exact(xout))];
bdCond = set_boundary!( mesh, xin, xout, mBdCond );

# create high mesh
cgMesh = aggmg.CgMesh( mesh, p );

# mesh operators and rhs
A, b = aggmg.cg_stiffness_and_rhs( cgMesh, mesh, func, bdCond );

# smoothers on mesh
smoother1 = aggmg.cg_smoother( cgMesh, A, :jac );
smoother2 = aggmg.cg_smoother( cgMesh, A, :addSchwarz );
smoother3 = aggmg.cg_smoother( cgMesh, A, :hybridSchwarz );

# create vector of x-values of nodes
x = zeros( cgMesh.mNumNodes );
for el in cgMesh.mElements
    x[ el.mNodesInd ] = el.mNodesX;
end
perm = sortperm(x);

s1 = ml.get_default_msession();

for i in 1:10
    u_fun(x) = sin(i*pi*x);
    u = u_fun.(x);

    uJac = u;
    uAdd = u;
    uHybrid = u;
    for j in 1:10
        uJac = uJac - aggmg.apply_smoother(smoother1, A*uJac; alpha = 2.0/3.0);
        uAdd = uAdd - aggmg.apply_smoother(smoother2, A*uAdd; alpha = 2.0/3.0);
        uHybrid = uHybrid - aggmg.apply_smoother(smoother3, A*uHybrid; alpha = 2.0/3.0);
    end

    ml.put_variable(s1, :x, x[perm]);
    ml.put_variable(s1, :u, u[perm]);
    ml.put_variable(s1, :uJac, uJac[perm]);
    ml.put_variable(s1, :uAdd, uAdd[perm]);
    ml.put_variable(s1, :uHybrid, uHybrid[perm]);
    ml.put_variable(s1, :i, i);

    ml.eval_string(s1, "figure(cast(i, 'like', 1));");
    ml.eval_string(s1, "plot(x, u, '-', 'LineWidth', 3); hold on;");
    ml.eval_string(s1, "plot(x, uJac, '-o', 'LineWidth', 2, 'MarkerSize', 10);");
    ml.eval_string(s1, "plot(x, uAdd, '-s', 'LineWidth', 2, 'MarkerSize', 10);");
    ml.eval_string(s1, "plot(x, uHybrid, '-*', 'LineWidth', 2, 'MarkerSize', 10);");
    ml.eval_string(s1, "legend('Original', 'Jacobi', 'Additive', 'Hybrid'); hold off;")
end

R1 = Matrix( la.I(size(A,1)) - aggmg.apply_smoother(smoother1, A; alpha = 2.0/3.0) );
R2 = Matrix( la.I(size(A,1)) - aggmg.apply_smoother(smoother2, A; alpha = 2.0/3.0) );
R3 = Matrix( la.I(size(A,1)) - aggmg.apply_smoother(smoother3, A; alpha = 2.0/3.0) );

S1, V1 = la.eigen(R1);
S2, V2 = la.eigen(R2);
S3, V3 = la.eigen(R3);

ml.eval_string(s1, "figure(11);");
ml.put_variable(s1, :S1, S1);
ml.put_variable(s1, :S2, real(S2));
ml.put_variable(s1, :S3, real(S3));
ml.eval_string(s1, "plot(S1, '-o', 'LineWidth', 3); hold on;");
ml.eval_string(s1, "plot(S2, '-s', 'LineWidth', 3); hold on;");
ml.eval_string(s1, "plot(S3, '-*', 'LineWidth', 3); hold on;");
ml.eval_string(s1, "legend('Jacobi', 'Additive', 'Hybrid'); hold off;")

ml.eval_string(s1, "figure(12);");
for i in size(V1,2)-3:size(V1,2)
    ml.put_variable(s1, :v, V1[:,i][perm]);
    ml.eval_string(s1, "plot(x, v, '-', 'LineWidth', 3); hold on;");
end
ml.eval_string(s1, "hold off;")

ml.eval_string(s1, "figure(13);");
for i in size(V2,2)-3:size(V2,2)
    ml.put_variable(s1, :v, real(V3[:,i][perm]));
    ml.eval_string(s1, "plot(x, v, '-', 'LineWidth', 3); hold on;");
end
ml.eval_string(s1, "hold off;")

ml.eval_string(s1, "figure(14);");
for i in size(V3,2)-3:size(V3,2)
    ml.put_variable(s1, :v, real(V3[:,i][perm]));
    ml.eval_string(s1, "plot(x, v, '-', 'LineWidth', 3); hold on;");
end
ml.eval_string(s1, "hold off;")