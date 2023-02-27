"""
    multigrid_v_cycle( H::MeshHierarchy, x0::AbstractVector, b::AbstractVector; 
    nPre::Integer = 3, nPost::Integer = 3, alpha::AbstractFloat = 2.0/3.0 )

Runs one `multigrid_v_cycle` for the mesh hierarchy `H` with initial guess `x0` 
and right hand side `b`.

# Arguments
- `H`: mesh hierarchy with operators for the Poisson equation
- `x0`: initial guess for the solution of the equation ``Ax = b``
- `b`: the right hand side of the equation ``Ax = b``
- `nPre = 3`: number of pre-smoothings
- `nPost = 3`: number of the post-smoothings
- `alpha = 2.0/3.0`: smoothing factor

# Outputs
- `x::AbstractVector`: updated solution after one `multigrid_v_cycle`
"""
function multigrid_v_cycle( H::MeshHierarchy, x0::AbstractVector, b::AbstractVector; 
    nPre::Integer = 3, nPost::Integer = 3, alpha::AbstractFloat = 2.0/3.0 )

    n = length( H.mMeshes );
    u = Vector{Vector{Float64}}( undef, n );
    rhs = Vector{Vector{Float64}}( undef, n );
    u[1] = x0;
    rhs[1] = b;

    for k = 1:n-1
        if k > 1
            u[k] = zeros( size( H.mStiffness[k], 2 ) );
        end
        for i = 1:nPre
            u[k] += apply_smoother( H.mSmoothers[k], rhs[k] - H.mStiffness[k] * u[k]; 
                alpha = alpha );
        end
        rhs[k+1] = H.mInterpolation[k]' * (rhs[k] - H.mStiffness[k] * u[k]);
    end

    u[n] = H.mStiffness[n] \ rhs[n];

    for k = n-1:-1:1
        u[k] += H.mInterpolation[k] * u[k+1];
        for i = 1:nPost
            u[k] += apply_smoother( H.mSmoothers[k], rhs[k] - H.mStiffness[k] * u[k]; 
                alpha = alpha );
        end
    end

    return u[1];
end

"""
    ldiv!( H::MeshHierarchy, b::AbstractVector )

Uses multigrid to solve the equation ``Ax = b``, where `H.meshes[1].A = A`. Replaces `b` 
with the solution.

# Arguments
- `H`: mesh hierarchy with operators for the Poisson equation
- `b`: the right hand side of the equation ``Ax = b`` we wish to solve. Overwritten with 
the solution
"""
function ldiv!( H::MeshHierarchy, b::AbstractVector )
    # initial guess of zeros for the solution
    u0 = zeros( size( H.mStiffness[1], 1 ) );

    # solve
    b[:] = multigrid_v_cycle( H, u0, b );

    return;
end

"""
    ldiv!( y::AbstractVector, H::MeshHierarchy, b::AbstractVector )

Uses multigrid to solve the equation ``Ax = b``, where `H.meshes[1].A = A`, and stores the 
solution in `y`.

# Arguments
- `y`: vector in which to store the solution
- `H`: mesh hierarchy with operators for the Poisson equation
- `b`: the right hand side of the equation ``Ax = b`` we wish to solve
"""
function ldiv!( y::AbstractVector, H::MeshHierarchy, b::AbstractVector )
    # initial guess of zeros for the solution
    u0 = zeros( size( H.mStiffness[1], 1 ) );

    # solve
    y[:] = multigrid_v_cycle( H, u0, b );

    return;
end

"""
    multigrid( H::MeshHierarchy, x0::AbstractVector, b::AbstractVector, maxiter::Integer, 
    tol::AbstractFloat )

Runs iterations of `multigrid_v_cycle` for the mesh hierarchy `H` with initial guess `x0` 
and right hand side `b` until tolerance `tol` is achieved in the residual or the maximum 
number of iterations `maxiter` is reached.

# Arguments
- `H`: mesh hierarchy with operators for the Poisson equation
- `x0`: initial guess for the solution of the equation ``Ax = b``
- `b`: the right hand side of the equation ``Ax = b``
- `maxiter`: maximum iterations of `multigrid_v_cycle` that will be run
- `tol`: desired tolerance to be achieved in the residual

# Outputs
- `x`: solution of the equation ``Ax = b``
- `iter::Integer`: number of iterations done
- `res::AbstractVector`: vector of residual of solution at every iteration
- `err::AbstractVector`: vector of error in solution compared to `\` at every 
iteration
"""
function multigrid( H::MeshHierarchy, x0::AbstractVector, b::AbstractVector, maxiter::Integer, 
    tol::AbstractFloat )

    x = zeros( length( x0 ) );
    u_exact = H.mStiffness[1] \ b;
    err = Vector{Float64}( undef, 0 );
    res = Vector{Float64}( undef, 0 );

    for i = 1:maxiter
        x = multigrid_v_cycle( H, x0, b );
        x0 = x;

        push!( err, la.norm( x - u_exact, 2 ) );
        push!( res, la.norm( H.mStiffness[1] * x - b, 2 ) );

        if res[i] < tol * la.norm( b, 2 )
            break;
        end
    end

    iter = length( res );

    return x, iter, res, err;
end


# """
#     jacobi( H::MeshHierarchy, x0::AbstractVector, b::AbstractVector, maxiter::Integer, 
#     tol::AbstractFloat )

# Runs iterations of the Jacobi iterative solver for the mesh hierarchy `H` with initial guess 
# `x0` and right hand side `b` until tolerance `tol` is achieved in the residual or the maximum 
# number of iterations `maxiter` is reached.

# # Arguments
# - `H`: mesh hierarchy with operators for the Poisson equation
# - `x0`: initial guess for the solution of the equation ``Ax = b``
# - `b`: the right hand side of the equation ``Ax = b``
# - `maxiter`: maximum iterations of `multigrid_v_cycle` that will be run
# - `tol`: desired tolerance to be achieved in the residual

# # Outputs
# - `x`: solution of the equation ``Ax = b``
# - `iter::Integer`: number of iterations done
# - `res::AbstractVector`: vector of residual of solution at every iteration
# - `err::AbstractVector`: vector of error in solution compared to `\` at every 
# iteration
# """
# function jacobi( H::MeshHierarchy, x0::AbstractVector, b::AbstractVector, maxiter::Integer, 
#     tol::AbstractFloat )

#     x = zeros( length( x0 ) );
#     u_exact = H.meshes[1].A \ b;
#     err = Vector{Float64}( undef, 0 );
#     res = Vector{Float64}( undef, 0 );

#     for i = 1:maxiter
#         x = x0 + H.meshes[1].smoother \ ( b - H.meshes[1].A * x0 );
#         x0 = x;

#         push!( err, la.norm( x - u_exact, 2 ) );
#         push!( res, la.norm( H.meshes[1].A * x - b, 2 ) );

#         if res[i] < tol * la.norm( b, 2 )
#             break;
#         end
#     end

#     iter = length( res );

#     return x, iter, res, err;
# end

function iterative_smoother_solve( A::AbstractMatrix, smoother, x0::AbstractVector, 
    b::AbstractVector; maxiter::Integer = 1000, tol::AbstractFloat = 1e-6, 
    alpha::AbstractFloat = 1.0 )

    x = zeros( length( x0 ) );
    uExact = A \ b;
    err = Vector{Float64}( undef, 0 );
    res = Vector{Float64}( undef, 0 );

    for i = 1:maxiter
        x = x0 + apply_smoother( smoother, b - A * x0; alpha = alpha );
        x0 = x;

        push!( err, la.norm( x - uExact, 2 ) );
        push!( res, la.norm( A * x - b, 2 ) );

        if res[i] < tol * la.norm( b, 2 )
            break;
        end
    end

    iter = length( res );

    return x, iter, res, err;
end