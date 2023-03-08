# """
#     MeshHierarchy

# Stores a vector of meshes along with boundary information for the Poisson problem

# # Fields
# - `mMeshes::Vector{AbstractMesh}`: vector of meshes
# - `mLBdFlag::Symbol`: flag for boundary condition at left boundary
# - `mRBdFlag::Symbol`: flag for boundary condition at right boundary
# - `mLDirVal::Float64`: left Dirichlet boundary condition
# - `mRDirVal::Float64`: right Dirichlet boundary condition
# - `mLNeumannVal::Float64`: left Neumann boundary condition
# - `mRNeumannVal::Float64`: right Neumann boundary condition
# - `mCDir::AbstractFloat`: Dirichlet penalty parameter
# - `mInterpFlag::Int64`: flag to determine which interpolation to use between CG and DG
# """
struct MeshHierarchy
    mMeshes::Vector{AbstractMesh}

    mStiffness::Vector{sp.SparseMatrixCSC{Float64, Int64}}
    mGradient::Vector{sp.SparseMatrixCSC{Float64, Int64}}
    mDivergence::Vector{sp.SparseMatrixCSC{Float64, Int64}}
    mC::Vector{sp.SparseMatrixCSC{Float64, Int64}}
    mSmoothers::Vector{AbstractSmoother}

    mInterpolation::Vector{AbstractMatrix{Float64}} # could also specify that they are sparse (don't use dense cg to dg)
    mBdConds::Vector{BoundaryCondition}
end

function MeshHierarchy( mMeshes, mesh, mBdConds, A; nCG = 1, nDG = 0, nAgg = 0, 
    CDir = 1.0 )

    if nCG <= 0 
        throw( ArgumentError( "At least one CG mesh required." ) );
    end
    if length( mMeshes ) != nCG + nDG + nAgg
        throw( ArgumentError( "Length of vector of meshes does not match inputed 
            number of CG, DG, and agglomerated meshes." ) );
    end

    mStiffness = Vector{sp.SparseMatrixCSC{Float64, Int64}}(undef, nCG + nDG + nAgg);

    mGradient = Vector{sp.SparseMatrixCSC{Float64, Int64}}(undef, nDG + nAgg);
    mDivergence = Vector{sp.SparseMatrixCSC{Float64, Int64}}(undef, nDG + nAgg);
    mC = Vector{sp.SparseMatrixCSC{Float64, Int64}}(undef, nDG + nAgg);
    
    mSmoothers = Vector{AbstractSmoother}(undef, nCG + nDG + nAgg);
    mInterpolation = Vector{AbstractMatrix{Float64}}(undef, nCG + nDG + nAgg - 1);

    mStiffness[1] = A;
    mSmoothers[1] = cg_smoother( mMeshes[1], A, :jac );

    for i = 1:(nCG-1)
        L = cg_cg_interpolation( mMeshes[i+1], mMeshes[i] );
        mInterpolation[i] = L;

        mStiffness[i+1] = L'*mStiffness[i]*L;
        mSmoothers[i+1] = cg_smoother( mMeshes[i+1], mStiffness[i+1], :jac );
    end

    if nDG >= 1
        mInterpolation[nCG] = dg_cg_interpolation( mMeshes[nCG+1], mMeshes[nCG], mesh, 
            1 );

        mGradient[1], mDivergence[1], mC[1] = dg_flux_operators( mMeshes[nCG+1], mesh, 
            mBdConds[nCG+1], CDir );
        # mGradient[1] = gradient( mMeshes[nCG+1], mesh, mBdConds[nCG+1] );
        # mDivergence[1] = divergence( mMeshes[nCG+1], mesh, mBdConds[nCG+1] );
        # mC[1] = C_matrix( mMeshes[nCG+1], mesh, mBdConds[nCG+1], CDir );

        mStiffness[nCG+1] = mC[1] - mDivergence[1] * 
            ( mMeshes[nCG+1].mMassMatrixLU \ mGradient[1] );
        mSmoothers[nCG+1] = dg_smoother( mMeshes[nCG+1], mStiffness[nCG+1], :blockJac );

        for i = 1:(nDG-1)
            L = dg_dg_interpolation( mMeshes[nCG+i+1], mMeshes[nCG+i] );
            mInterpolation[nCG+i] = L;

            mGradient[i+1] = L'*mGradient[i]*L;
            mDivergence[i+1] = L'*mDivergence[i]*L;
            mC[i+1] = L'*mC[i]*L;

            mStiffness[nCG+i+1] = mC[i+1] - mDivergence[i+1] * 
                ( mMeshes[nCG+i+1].mMassMatrixLU \ mGradient[i+1] );
            mSmoothers[nCG+i+1] = dg_smoother( mMeshes[nCG+i+1], mStiffness[nCG+i+1],
                :blockJac );
        end

        for i = 0:(nAgg-1)
            if i == 0
                L = aggdg_dg_interpolation( mMeshes[nCG+nDG+i+1], mMeshes[nCG+nDG+i] );
            else
                L = aggdg_aggdg_interpolation( mMeshes[nCG+nDG+i+1], mMeshes[nCG+nDG+i], 
                    mMeshes[nCG+nDG] );
            end
            mInterpolation[nCG+nDG+i] = L;

            mGradient[nDG+i+1] = L'*mGradient[nDG+i]*L;
            mDivergence[nDG+i+1] = L'*mDivergence[nDG+i]*L;
            mC[nDG+i+1] = L'*mC[nDG+i]*L;

            mStiffness[nCG+nDG+i+1] = mC[nDG+i+1] - mDivergence[nDG+i+1] * 
                ( mMeshes[nCG+nDG+i+1].mMassMatrixLU \ mGradient[nDG+i+1] );
            mSmoothers[nCG+nDG+i+1] = dg_smoother( mMeshes[nCG+nDG+i+1], 
                mStiffness[nCG+nDG+i+1], :blockJac );
        end
    elseif nDG == 0
        if nAgg >= 1
            mInterpolation[nCG] = aggdg_cg_interpolation( mMeshes[nCG+1], mMeshes[nCG], 
                mesh, 1 );

            mGradient[1], mDivergence[1], mC[1] = dg_flux_operators( mMeshes[nCG+1], 
                mMeshes[nCG], mBdConds[nCG+1], CDir );

            mStiffness[nCG+1] = mC[1] - mDivergence[1] * 
                ( mMeshes[nCG+1].mMassMatrixLU \ mGradient[1] );
            mSmoothers[nCG+1] = dg_smoother( mMeshes[nCG+1], mStiffness[nCG+1], :blockJac );

            for i = 1:(nAgg-1)
                L = aggdg_aggdg_interpolation( mMeshes[nCG+i+1], mMeshes[nCG+i], 
                    mMeshes[nCG] );
                mInterpolation[nCG+i] = L;
    
                mGradient[i+1] = L'*mGradient[i]*L;
                mDivergence[i+1] = L'*mDivergence[i]*L;
                mC[i+1] = L'*mC[i]*L;
    
                mStiffness[nCG+i+1] = mC[i+1] - mDivergence[i+1] * 
                    ( mMeshes[nCG+i+1].mMassMatrixLU \ mGradient[i+1] );
                mSmoothers[nCG+i+1] = dg_smoother( mMeshes[nCG+i+1], mStiffness[nCG+i+1], 
                    :blockJac );
            end
        end
    end

    return MeshHierarchy( mMeshes, mStiffness, mGradient, mDivergence, mC, mSmoothers, 
        mInterpolation, mBdConds );
end

function MeshHierarchy( mMeshes, mBdConds, A, G, D, C; nDG = 1, nAgg = 0 )

    if nDG <= 0 
        throw( ArgumentError( "At least one DG mesh required." ) );
    end
    if length( mMeshes ) != nDG + nAgg
        throw( ArgumentError( "Length of vector of meshes does not match inputed 
            number of DG and agglomerated meshes." ) );
    end

    mStiffness = Vector{sp.SparseMatrixCSC{Float64, Int64}}(undef, nDG + nAgg);

    mGradient = Vector{sp.SparseMatrixCSC{Float64, Int64}}(undef, nDG + nAgg);
    mDivergence = Vector{sp.SparseMatrixCSC{Float64, Int64}}(undef, nDG + nAgg);
    mC = Vector{sp.SparseMatrixCSC{Float64, Int64}}(undef, nDG + nAgg);
    
    mSmoothers = Vector{AbstractSmoother}(undef, nDG + nAgg);
    mInterpolation = Vector{AbstractMatrix{Float64}}(undef, nDG + nAgg - 1);

    mGradient[1] = G;
    mDivergence[1] = D;
    mC[1] = C;

    mStiffness[1] = A;
    mSmoothers[1] = dg_smoother( mMeshes[1], A, :blockJac );

    for i = 1:(nDG-1)
        L = dg_dg_interpolation( mMeshes[i+1], mMeshes[i] );
        mInterpolation[i] = L;

        mGradient[i+1] = L'*mGradient[i]*L;
        mDivergence[i+1] = L'*mDivergence[i]*L;
        mC[i+1] = L'*mC[i]*L;

        mStiffness[i+1] = mC[i+1] - mDivergence[i+1] * 
            ( mMeshes[i+1].mMassMatrixLU \ mGradient[i+1] );
        mSmoothers[i+1] = dg_smoother( mMeshes[i+1], mStiffness[i+1], :blockJac );
    end

    return MeshHierarchy( mMeshes, mStiffness, mGradient, mDivergence, mC, mSmoothers, 
        mInterpolation, mBdConds );
end