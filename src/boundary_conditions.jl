struct BoundaryCondition
    mBdCond::Vector{ Tuple{Symbol, Float64} }
    mDirNodes::Vector{Int64}
    mDirVals::Vector{Float64}
    mNeuNodes::Vector{Int64}
end