struct BlockDiagonal{T, MT<:AbstractMatrix{T}} <: AbstractMatrix{T}
    mBlocks::Vector{MT}
    mBlockInds::Vector{Tuple{Int64, Int64}}

    function BlockDiagonal( mBlocks)
        mBlockInds = Vector{Tuple{Int64, Int64}}( undef, length(mBlocks) );

        counter = 0
        for (j, block) in enumerate(mBlocks)
            mBlockInds[j] = (counter + 1, counter + size(mBlocks, 1));


        return new{eltype(first(mBlocks)),eltype(mBlocks)}( mBlocks, mBlockInds );
    end
end

