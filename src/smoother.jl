struct AdditiveSchwarzSmoother <: AbstractSmoother
    mBlocks::Vector{la.LU{Float64, Matrix{Float64}}}
    mBlockInds::Matrix{Int64}
end

function apply_smoother(A::AdditiveSchwarzSmoother, B::AbstractVecOrMat; 
    alpha::Float64 = 1.0)

    Y = zeros( size(B) );

    for j = 1:size(B,2)
        for (i, block) in enumerate(A.mBlocks)
            Y[ A.mBlockInds[:,i], j ] += block \ B[ A.mBlockInds[:,i], j ];
        end
    end

    return alpha * Y;
end

############################################################################################
############################################################################################
############################################################################################

struct HybridSchwarzSmoother <: AbstractSmoother
    mBlocks::Vector{la.LU{Float64, Matrix{Float64}}}
    mBlockInds::Matrix{Int64}
    mCountingMatrix::la.Diagonal{Float64}
end

function apply_smoother(A::HybridSchwarzSmoother, B::AbstractVecOrMat;
    alpha::Float64 = 1.0)

    Y = zeros( size(B) );

    for j = 1:size(B,2)
        temp = zeros( size(B,1) );

        for (i, block) in enumerate(A.mBlocks)
            temp[ A.mBlockInds[:,i] ] += block \ B[ A.mBlockInds[:,i], j ];
        end

        Y[:, j] = A.mCountingMatrix \ temp;
    end

    return alpha * Y;
end

############################################################################################
############################################################################################
############################################################################################

struct JacobiSmoother <: AbstractSmoother
    mJac::la.Diagonal{Float64}
end

function apply_smoother(A::JacobiSmoother, B::AbstractVecOrMat; alpha::Float64 = 1.0)
    return alpha * (A.mJac \ B);
end

############################################################################################
############################################################################################
############################################################################################

struct BlockJacobi <: AbstractSmoother
    mBlocks::Vector{la.LU{Float64, Matrix{Float64}}}
    mBlockInds::Matrix{Int64}
end

function apply_smoother(A::BlockJacobi, B::AbstractVecOrMat; 
    alpha::Float64 = 1.0)

    Y = zeros( size(B) );

    for j = 1:size(B,2)
        for (i, block) in enumerate(A.mBlocks)
            Y[ A.mBlockInds[:,i], j ] += block \ B[ A.mBlockInds[:,i], j ];
        end
    end

    return alpha * Y;
end

############################################################################################
############################################################################################
############################################################################################

# function to initialize mSmoother for a CgMesh and a stiffness matrix A
function cg_smoother( cgMesh::CgMesh, A::sp.SparseMatrixCSC{Float64, Int64}, 
    smootherType::Symbol )
    # for Jacobi, all I need is A
    # for block Schwarz, I need to know vertices in each element and their local and global. 
    # I also need to know how many faces each vertex is in. This should all be given by the 
    # cgMesh.

    if smootherType == :jac

        jac = la.Diagonal{Float64}( undef, size(A,1) );
        for i = 1:size(A,1)
            jac[i,i] = A[i,i];
        end

        smoother = JacobiSmoother( jac );

    elseif smootherType == :addSchwarz

        n = length( cgMesh.mElements );
        p = cgMesh.mP;
        blocks = Vector{la.LU{Float64,Matrix{Float64}}}( undef, n );
        blockInds = zeros( Int64, p+1, n );

        for (i, el) in enumerate( cgMesh.mElements )
            blocks[i] = la.lu( Matrix( A[ el.mNodesInd, el.mNodesInd ] ) );
            blockInds[:,i] = el.mNodesInd;
        end

        smoother = AdditiveSchwarzSmoother( blocks, blockInds );

    elseif smootherType == :hybridSchwarz

        n = length( cgMesh.mElements );
        p = cgMesh.mP;
        blocks = Vector{la.LU{Float64,Matrix{Float64}}}( undef, n );
        blockInds = zeros( Int64, p+1, n );
        countingMatrix = la.Diagonal{Float64}( zeros( size(A,1) ) );

        for (i, el) in enumerate( cgMesh.mElements )
            blocks[i] = la.lu( Matrix( A[ el.mNodesInd, el.mNodesInd ] ) );
            blockInds[:,i] = el.mNodesInd;
            for l in el.mNodesInd
                countingMatrix[l,l] += 1.0;
            end
        end

        smoother = HybridSchwarzSmoother( blocks, blockInds, countingMatrix );

    end

    return smoother;
end

# function to initialize mSmoother for a DgMesh and a stiffness matrix A
function dg_smoother( dgMesh::Union{DgMesh, AgglomeratedDgMesh1, AgglomeratedDgMeshN}, 
    A::sp.SparseMatrixCSC{Float64, Int64}, smootherType::Symbol )
    # for block Jacobi, I need to know vertices in each element and A

    if smootherType == :jac
        jac = la.Diagonal{Float64}( undef, size(A,1) );
        for i = 1:size(A,1)
            jac[i,i] = A[i,i];
        end

        smoother = JacobiSmoother( jac );
    elseif smootherType == :blockJac
        n = length( dgMesh.mElements );
        p = dgMesh.mP;
        blocks = Vector{la.LU{Float64,Matrix{Float64}}}( undef, n );
        blockInds = zeros( Int64, p+1, n );

        for (i, el) in enumerate( dgMesh.mElements )
            blocks[i] = la.lu( Matrix( A[ el.mNodesInd, el.mNodesInd ] ) );
            blockInds[:,i] = el.mNodesInd;
        end

        smoother = BlockJacobi( blocks, blockInds );
    end

    return smoother;
end