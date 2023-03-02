import LinearAlgebra as la
import SparseArrays as sp
import SuiteSparse

import Base: *, \
import Base: size, similar
import LinearAlgebra: mul!, ldiv!, lu

include("../src/blockdiagonal.jl")

A = BlockDiagonal( [rand(3,3) for i in 1:3] );

B = sp.sprandn( 9, 9, 0.3 );

rowValCol, nzValCol = bd_sp_colmul( A, B, 1 );

A2 = zeros( size(A) );
A2[1:3,1:3] = A.mBlocks[1];
A2[4:6,4:6] = A.mBlocks[2];
A2[7:9,7:9] = A.mBlocks[3];

d2 = A2 * B[:,1];
d = zeros( length(d2) );
d[rowValCol] = nzValCol;

println(la.norm(d - d2));

C = A * B;
C2 = A2 * B;
println(la.norm(C - C2));

b = B[:, 4];
c = A * b;
c2 = A2 * b;
println(la.norm(c - c2));

ALU = la.lu( A );

F = ALU \ B;
F2 = A2 \ Array(B);
println(la.norm(F - F2))

f = ALU \ b
f2 = A2 \ b
println(la.norm(f - f2))