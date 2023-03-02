const DenseMatrixUnion = Union{ StridedMatrix, la.LowerTriangular, la.UnitLowerTriangular, 
    la.UpperTriangular, la.UnitUpperTriangular, BitMatrix }
const AdjOrTransDenseMatrix = Union{ DenseMatrixUnion, la.Adjoint{<:Any, <:DenseMatrixUnion},
    la.Transpose{<:Any, <:DenseMatrixUnion} }
const DenseInputVector = Union{StridedVector, BitVector}
const DenseInputVecOrMat = Union{AdjOrTransDenseMatrix, DenseInputVector}
const SparseMatrixCSCUnion = sp.SparseMatrixCSCUnion
const SparseOrTri = Union{ sp.SparseMatrixCSCUnion, sp.SparseTriangular }
const SparseMatOrVec = Union{ sp.SparseOrTri, sp.SparseVectorUnion, 
    SubArray{<:Any,<:Any,<:sp.AbstractSparseArray} }

struct BlockDiagonal # <: AbstractMatrix
    mBlocks::Vector{Matrix{Float64}}
    mBlockSize::Int64
    mBlockInds::Matrix{Int64}
end

struct BlockDiagonalLU # <: AbstractMatrix
    mBlocksLU::Vector{la.LU{Float64, Matrix{Float64}}}
    mBlockSize::Int64
    mBlockInds::Matrix{Int64}
end

############################################################################################
# creation of BlockDiagonal
############################################################################################

function BlockDiagonal( mBlocks )

    mBlockSize = size( mBlocks[1], 1 );
    mBlockInds = zeros( mBlockSize, length(mBlocks) );

    for (j, block) in enumerate( mBlocks )
        mBlockInds[:,j] = ((j-1) * mBlockSize + 1):j*mBlockSize;

        if size(block) != (mBlockSize, mBlockSize)
            throw( ArgumentError( "All blocks must be of the same size." ) );
        end
    end

    return BlockDiagonal( mBlocks, mBlockSize, mBlockInds );
end

############################################################################################
# creation of BlockDiagonalLU
############################################################################################

function BlockDiagonalLU( A::BlockDiagonal )

    mBlockSize = A.mBlockSize
    mBlocksLU = Vector{la.LU{Float64, Matrix{Float64}}}( undef, length(A.mBlocks) );
    mBlockInds = A.mBlockInds;

    for (j, block) in enumerate( A.mBlocks )
        mBlocksLU[j] = la.lu( block );
    end

    return BlockDiagonalLU( mBlocksLU, mBlockSize, mBlockInds );
end

function BlockDiagonalLU( blocks::Vector{Matrix{Float64}} )

    mBlockSize = size( blocks[1], 1 );
    mBlocksLU = Vector{la.LU{Float64, Matrix{Float64}}}( undef, length(blocks) );
    mBlockInds = zeros( mBlockSize, length(mBlocks) );

    for (j, block) in enumerate( blocks )
        mBlocksLU[j] = la.lu( block );
        mBlockInds[:,j] = ((j-1) * mBlockSize + 1):j*mBlockSize;

        if size(block) != (mBlockSize, mBlockSize)
            throw( ArgumentError( "All blocks must be of the same size." ) );
        end
    end

    return BlockDiagonalLU( mBlocksLU, mBlockSize, mBlockInds );
end

############################################################################################
# utility for BlockDiagonal
############################################################################################

size(A::BlockDiagonal) = ( length(A.mBlocks) * A.mBlockSize, 
                           length(A.mBlocks) * A.mBlockSize );

function size(A::BlockDiagonal, i::Integer)
    if i <= 0
        error( "arraysize: dimension out of range" )
    elseif i >= 3
        return 1;
    else
        return length(A.mBlocks) * A.mBlockSize;
    end
end

function similar(A::BlockDiagonal)
    mBlocks = Vector{Matrix{Float64}}(undef, length(A.mBlocks));
    mBlockSize = A.mBlockSize
    mBlockInds = A.mBlockInds

    for (j, block) in enumerate( A.mBlocks )
        mBlocks[j] = similar( block );
    end

    return BlockDiagonal( mBlocks, mBlockSize, mBlockInds );
end

############################################################################################
# Linear Algebra for BlockDiagonal
############################################################################################

function mul!( C::BlockDiagonal, A::BlockDiagonal, b::Number)
    size(A) == size(C) || throw(DimensionMismatch())
    length( A.mBlocks ) == length( C.mBlocks ) || error("C and A must have the same number of blocks.")
    A.mBlockSize == A.mBlockSize || error("C and A must have the same blocksize.")

    C.mBlockInds[:,:] = A.mBlockInds;
    for (j, block) in enumerate( A.mBlocks )
        C.mBlocks[j][:,:] = b * block
    end

    return C;
end

function mul!( C::BlockDiagonal, b::Number, A::BlockDiagonal)
    size(A) == size(C) || throw(DimensionMismatch())
    length( A.mBlocks ) == length( C.mBlocks ) || error("C and A must have the same number of blocks.")
    A.mBlockSize == A.mBlockSize || error("C and A must have the same blocksize.")

    C.mBlockInds[:,:] = A.mBlockInds;
    for (j, block) in enumerate( A.mBlocks )
        C.mBlocks[j][:,:] = b * block
    end

    return C;
end

*( A::BlockDiagonal, b::Number ) = mul!( similar(A), A, b );
*( b::Number, A::BlockDiagonal ) = mul!( similar(A), b, A );

function mul!( C::StridedVecOrMat, A::BlockDiagonal, B::DenseInputVecOrMat )
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())

    for (j, block) in enumerate( A.mBlocks )
        C[ A.mBlockInds[:,j], : ] += block * B[ A.mBlockInds[:,j], : ];
    end

    return C
end

*( A::BlockDiagonal, x::DenseInputVector ) = mul!( zeros( length(x) ), A, x );
*( A::BlockDiagonal, B::AdjOrTransDenseMatrix ) = mul!( zeros( size(B) ), A, B );

function mul!( C::StridedVecOrMat, A::AdjOrTransDenseMatrix, B::BlockDiagonal )
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())

    for (j, block) in enumerate( B.mBlocks )
        C[ :, B.mBlockInds[:,j] ] += A[ :, B.mBlockInds[:,j] ] * block;
    end

    return C
end

*( A::AdjOrTransDenseMatrix, B::BlockDiagonal ) = mul!( zeros( size(A) ), A, B );

function bd_sp_matmul( A::BlockDiagonal, B::SparseMatOrVec )
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    colPtrC = Vector{Int64}( undef, size(B, 2) + 1 );
    rowValC = Vector{Int64}( undef, 0 );
    nzValC = Vector{Float64}( undef, 0 );

    colPtrC[1] = 1;

    for col in 1:size(B, 2)
        rowValCol, nzValCol = bd_sp_colmul( A, B, col );

        append!( rowValC, rowValCol );
        append!( nzValC, nzValCol );
        colPtrC[col+1] = colPtrC[col] + length( rowValCol );
    end

    C = sp.SparseMatrixCSC( size(A, 1), size(B, 2), colPtrC, rowValC, nzValC );

    return C;
end

function bd_sp_colmul( A::BlockDiagonal, B::SparseMatOrVec, col::Int64 )
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    nzIndB = sp.nzrange(B, col);
    nzValB = sp.nonzeros(B)[nzIndB];
    rowValB = sp.rowvals(B)[nzIndB];

    rowValCol = Vector{Int64}(undef, 0);
    nzValCol = Vector{Float64}(undef, 0);

    ind = 1;
    while ind <= length( nzValB )
        row = rowValB[ind];
        blockInd = div( row - 1, A.mBlockSize ) + 1;
        minBlockRow = A.mBlockInds[ 1, blockInd ];
        maxBlockRow = A.mBlockInds[ end, blockInd ];

        tempVec = zeros( A.mBlockSize );
        tempVec[ row - minBlockRow + 1] = nzValB[ind];

        j = 1;
        if ind + j <= length( nzValB )
            tempRow = rowValB[ind + j];
        else
            tempRow = row + A.mBlockSize;
        end

        while tempRow <= maxBlockRow
            tempVec[ tempRow - minBlockRow + 1 ] = nzValB[ind + j];
            j += 1;
            if ind + j <= length( nzValB )
                tempRow = rowValB[ind + j];
            else
                tempRow = row + A.mBlockSize;
            end
        end
        ind += j;

        b = A.mBlocks[ blockInd ] * tempVec;

        for i in 1:A.mBlockSize
            push!( rowValCol, minBlockRow + i - 1 );
            push!( nzValCol, b[i] );
        end
    end

    return rowValCol, nzValCol
end

*( A::BlockDiagonal, B::sp.AbstractSparseVector ) = bd_sp_matmul( A, B )[:,1];
*( A::BlockDiagonal, B::sp.SparseColumnView ) = bd_sp_matmul( A, B )[:,1];
*( A::BlockDiagonal, B::sp.SparseVectorView ) = bd_sp_matmul( A, B )[:,1];
*( A::BlockDiagonal, B::SparseMatrixCSCUnion ) = bd_sp_matmul( A, B );
*( A::BlockDiagonal, B::sp.SparseTriangular ) = bd_sp_matmul( A, B );
*( A::BlockDiagonal, B::la.Adjoint{<:Any,<:sp.AbstractSparseMatrixCSC} ) = bd_sp_matmul( A, 
    copy(B) );
*( A::BlockDiagonal, B::la.Transpose{<:Any,<:sp.AbstractSparseMatrixCSC} ) = 
    bd_sp_matmul( A, copy(B) );

lu( A::BlockDiagonal ) = BlockDiagonalLU( A );

############################################################################################
# utility for BlockDiagonalLU
############################################################################################

size(A::BlockDiagonalLU) = ( length(A.mBlocksLU) * A.mBlockSize, 
                             length(A.mBlocksLU) * A.mBlockSize );

function size(A::BlockDiagonalLU, i::Integer)
    if i <= 0
        error( "arraysize: dimension out of range" )
    elseif i >= 3
        return 1;
    else
        return length(A.mBlocksLU) * A.mBlockSize;
    end
end

############################################################################################
# Linear Algebra for BlockDiagonalLU
############################################################################################

function bd_sp_solve( A::BlockDiagonalLU, B::SparseMatOrVec )
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    colPtrC = Vector{Int64}( undef, size(B, 2) + 1 );
    rowValC = Vector{Int64}( undef, 0 );
    nzValC = Vector{Float64}( undef, 0 );

    colPtrC[1] = 1;

    for col in 1:size(B, 2)
        rowValCol, nzValCol = bd_sp_colsolve( A, B, col );

        append!( rowValC, rowValCol );
        append!( nzValC, nzValCol );
        colPtrC[col+1] = colPtrC[col] + length( rowValCol );
    end

    C = sp.SparseMatrixCSC( size(A, 1), size(B, 2), colPtrC, rowValC, nzValC );

    return C;
end

function bd_sp_colsolve( A::BlockDiagonalLU, B::SparseMatOrVec, col::Int64 )
    size(A, 2) == size(B, 1) || throw(DimensionMismatch())

    nzIndB = sp.nzrange(B, col);
    nzValB = sp.nonzeros(B)[nzIndB];
    rowValB = sp.rowvals(B)[nzIndB];

    rowValCol = Vector{Int64}(undef, 0);
    nzValCol = Vector{Float64}(undef, 0);

    ind = 1;
    while ind <= length( nzValB )
        row = rowValB[ind];
        blockInd = div( row - 1, A.mBlockSize ) + 1;
        minBlockRow = A.mBlockInds[ 1, blockInd ];
        maxBlockRow = A.mBlockInds[ end, blockInd ];

        tempVec = zeros( A.mBlockSize );
        tempVec[ row - minBlockRow + 1] = nzValB[ind];

        j = 1;
        if ind + j <= length( nzValB )
            tempRow = rowValB[ind + j];
        else
            tempRow = row + A.mBlockSize;
        end

        while tempRow <= maxBlockRow
            tempVec[ tempRow - minBlockRow + 1 ] = nzValB[ind + j];
            j += 1;
            if ind + j <= length( nzValB )
                tempRow = rowValB[ind + j];
            else
                tempRow = row + A.mBlockSize;
            end
        end
        ind += j;

        b = A.mBlocksLU[ blockInd ] \ tempVec;

        for i in 1:A.mBlockSize
            push!( rowValCol, minBlockRow + i - 1 );
            push!( nzValCol, b[i] );
        end
    end

    return rowValCol, nzValCol
end

\( A::BlockDiagonalLU, B::sp.AbstractSparseVector ) = bd_sp_solve( A, B )[:,1];
\( A::BlockDiagonalLU, B::sp.SparseColumnView ) = bd_sp_solve( A, B )[:,1];
\( A::BlockDiagonalLU, B::sp.SparseVectorView ) = bd_sp_solve( A, B )[:,1];
\( A::BlockDiagonalLU, B::SparseMatrixCSCUnion ) = bd_sp_solve( A, B );
\( A::BlockDiagonalLU, B::sp.SparseTriangular ) = bd_sp_solve( A, B );
\( A::BlockDiagonalLU, B::la.Adjoint{<:Any,<:sp.AbstractSparseMatrixCSC} ) = bd_sp_solve( A, 
    copy(B) );
\( A::BlockDiagonalLU, B::la.Transpose{<:Any,<:sp.AbstractSparseMatrixCSC} ) = 
    bd_sp_solve( A, copy(B) );