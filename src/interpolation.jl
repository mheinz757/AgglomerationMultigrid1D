############################################################################################
# interpolation between CgMeshes
############################################################################################

function cg_cg_interpolation( lowMesh::CgMesh, highMesh::CgMesh )

    lowBasisFunVal = evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
        highMesh.mRefEl.mNodesX );

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    for (k, lowEl) in enumerate(lowMesh.mElements)
        highEl = highMesh.mElements[k];

        for j = 1:length( lowEl.mNodesInd )
            lowNode = lowEl.mNodesInd[j];
            for i = 3:length( highEl.mNodesInd )
                highNode = highEl.mNodesInd[i];
                push!( data, ( highNode, lowNode, 0.0) );
            end
        end

        for j = 1:2
            lowNode = lowEl.mNodesInd[j];
            highNode = highEl.mNodesInd[j];

            push!( data, ( highNode, lowNode, 0.0) );
        end
    end

    L = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        highMesh.mNumNodes, lowMesh.mNumNodes );

    for (k, lowEl) in enumerate(lowMesh.mElements)
        highEl = highMesh.mElements[k];

        for j = 1:length( lowEl.mNodesInd )
            lowNode = lowEl.mNodesInd[j];
            for i = 3:length( highEl.mNodesInd )
                highNode = highEl.mNodesInd[i];
                if abs(L[highNode, lowNode]) <= eps(Float64)
                    L[highNode, lowNode] = lowBasisFunVal[i,j];
                end
            end
        end

        for j = 1:2
            lowNode = lowEl.mNodesInd[j];
            highNode = highEl.mNodesInd[j];

            L[highNode, lowNode] = lowBasisFunVal[j,j];
        end
    end

    return L;
end

function cg_cg_interpolation2( lowMesh::CgMesh, highMesh::CgMesh )
    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    gaussQuadNodes, gaussQuadWeights = gauss_quad(lowMesh.mP + highMesh.mP);

    highBasisGQFunVal = evaluate_nodal_basis_fun( highMesh.mRefEl.mBasisFunCoeff, 
        gaussQuadNodes );
    lowBasisGQFunVal = evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
        gaussQuadNodes );

    for (k, lowEl) in enumerate(lowMesh.mElements)
        highEl = highMesh.mElements[k];

        temp = zeros( length(highEl.mNodesInd), length(lowEl.mNodesInd) )
        for j = 1:length(lowEl.mNodesInd), i = 1:length(highEl.mNodesInd), l = 1:length(gaussQuadNodes)
            temp[i,j] += lowEl.mJacobian * gaussQuadWeights[l] * highBasisGQFunVal[l,i] * 
                lowBasisGQFunVal[l, j];
        end

        for (j, lowNode) in enumerate(lowEl.mNodesInd), (i, highNode) in enumerate( highEl.mNodesInd)
            push!( data, ( highNode, lowNode, temp[i,j] ) );
        end
    end

    N = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        highMesh.mNumNodes, lowMesh.mNumNodes );

    return highMesh.mMassMatrixLU \ Array(N);
end

############################################################################################
# interpolation between DgMeshes
############################################################################################

function dg_dg_interpolation( lowMesh::DgMesh, highMesh::DgMesh )

    lowBasisFunVal = evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
        highMesh.mRefEl.mNodesX );

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    for (k, lowEl) in enumerate(lowMesh.mElements)
        highEl = highMesh.mElements[k];

        for (j, lowNode) in enumerate( lowEl.mNodesInd ), (i, highNode) in enumerate( highEl.mNodesInd )
            push!( data, ( highNode, lowNode, lowBasisFunVal[i,j] ) );
        end
    end

    L = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        highMesh.mNumNodes, lowMesh.mNumNodes );

    return L;
end

function dg_dg_interpolation2( lowMesh::DgMesh, highMesh::DgMesh )

    lowBasisFunVal = evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
        highMesh.mRefEl.mNodesX );

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    for (k, lowEl) in enumerate(lowMesh.mElements)
        highEl = highMesh.mElements[k];

        for (j, lowNode) in enumerate( lowEl.mNodesInd )
            for i = 3:length( highEl.mNodesInd )
                highNode = highEl.mNodesInd[i];
                push!( data, ( highNode, lowNode, lowBasisFunVal[i,j] ) );
            end
        end

        for j = 1:2
            lowNode = lowEl.mNodesInd[j];
            highNode = highEl.mNodesInd[j];

            push!( data, ( highNode, lowNode, lowBasisFunVal[j,j] ) );
        end
    end

    L = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        highMesh.mNumNodes, lowMesh.mNumNodes );

    return L;
end

############################################################################################
# interpolation between DgMesh and CgMesh
############################################################################################

function dg_cg_interpolation( lowMesh::DgMesh, highMesh::CgMesh, mesh::Mesh, 
    interpFlag::Int64 )

    if interpFlag == 0 || interpFlag == 1
        data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

        gaussQuadNodes, gaussQuadWeights = gauss_quad(lowMesh.mP + highMesh.mP);

        highBasisGQFunVal = evaluate_nodal_basis_fun( highMesh.mRefEl.mBasisFunCoeff, 
            gaussQuadNodes );
        lowBasisGQFunVal = evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
            gaussQuadNodes );

        for (k, lowEl) in enumerate(lowMesh.mElements)
            highEl = highMesh.mElements[k];

            temp = zeros( length(highEl.mNodesInd), length(lowEl.mNodesInd) );
            for j = 1:length(lowEl.mNodesInd), i = 1:length(highEl.mNodesInd), l = 1:length(gaussQuadNodes)
                temp[i,j] += lowEl.mJacobian * gaussQuadWeights[l] * 
                    highBasisGQFunVal[l,i] * lowBasisGQFunVal[l, j];
            end

            for (j, lowNode) in enumerate(lowEl.mNodesInd), (i, highNode) in enumerate( highEl.mNodesInd)
                push!( data, ( highNode, lowNode, temp[i,j] ) );
            end
        end

        N = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
            highMesh.mNumNodes, lowMesh.mNumNodes );
    elseif interpFlag == 2
        lowBasisFunVal = evaluate_nodal_basis_fun( lowMesh.mRefEl.mBasisFunCoeff, 
            highMesh.mRefEl.mNodesX );

        data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
        for (k, lowEl) in enumerate(lowMesh.mElements)
            highEl = highMesh.mElements[k];

            for j = 1:length( lowEl.mNodesInd )
                lowNode = lowEl.mNodesInd[j];
                
                for i = 1:2
                    highNode = highEl.mNodesInd[i];
                    vert = mesh.mVertices[highNode];
                    if isBoundary( vert )
                        push!( data, (highNode, lowNode, lowBasisFunVal[i,j] ) );
                    else
                        push!( data, (highNode, lowNode, 0.5*lowBasisFunVal[i,j] ) );
                    end
                end

                for i = 3:length( highEl.mNodesInd )
                    highNode = highEl.mNodesInd[i];
                    push!( data, ( highNode, lowNode, lowBasisFunVal[i,j] ) );
                end
            end
        end

        L = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
            highMesh.mNumNodes, lowMesh.mNumNodes );
    else
        throw( ArgumentError( "Only implemented for interpFlag = 0, 1, or 2." ) );
    end

    if interpFlag == 0
        L = highMesh.mMassMatrixLU \ Array(N);
    elseif interpFlag == 1
        lumpedMassMatrixVec = zeros( highMesh.mNumNodes );
        for j = 1:highMesh.mNumNodes
            lumpedMassMatrixVec[j] = sum( highMesh.mMassMatrix[j,:] );
        end

        L = la.Diagonal( lumpedMassMatrixVec ) \ N;
    end

    return L;
end

############################################################################################
# interpolation between two AgglomeratedDgMeshes
##########################################################################################

function aggdg_aggdg_interpolation( coarseMesh::AgglomeratedDgMeshN, 
    fineMesh::Union{AgglomeratedDgMesh1, AgglomeratedDgMeshN}, 
    baseMesh::Union{CgMesh, DgMesh} )

    if coarseMesh.mP != fineMesh.mP
        throw( ArgumentError( "The two agglomerated meshes must have the same p." ) );
        # could maybe do for different p (since p = 0 has constant basis function value)
    end

    p = coarseMesh.mP;

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    for coarseEl in coarseMesh.mElements
        count = 0;
        for fineElInd in coarseEl.mSubAggElementInds
            fineEl = fineMesh.mElements[ fineElInd ];
            temp = zeros( p+1, p+1 );

            for (k, baseElInd) in enumerate( fineEl.mBaseElementInds )
                baseEl = baseMesh.mElements[ baseElInd ];

                for j = 1:length( coarseEl.mNodesInd ), i = 1:length( fineEl.mNodesInd ), l = 1:length(fineMesh.mGaussQuadNodes)
                    temp[i,j] += baseEl.mJacobian * fineMesh.mGaussQuadWeights[l] * 
                        fineEl.mBasisGQFunVal[k][l,i] * coarseEl.mBasisGQFunVal[count+k][l,j];
                end
            end
            count += length( fineEl.mBaseElementInds );

            for (j, node2) in enumerate( coarseEl.mNodesInd ), (i, node1) in enumerate( fineEl.mNodesInd ) 
                push!(data, ( node1, node2, temp[i,j] ) );
            end
        end
    end

    N = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        fineMesh.mNumNodes, coarseMesh.mNumNodes );

    return sp.sparse( fineMesh.mMassMatrixLU \ Array(N) );
end

############################################################################################
# interpolation between AgglomeratedDgMesh and DgMesh
############################################################################################

function aggdg_dg_interpolation( aggMesh::Union{AgglomeratedDgMesh1, AgglomeratedDgMeshN}, 
    baseMesh::DgMesh )

    refEl = baseMesh.mRefEl;

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    for aggEl in aggMesh.mElements
        for (k, baseElInd) in enumerate( aggEl.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            aggBasisFunVal = evaluate_local_modal_basis_fun( aggMesh.mP, 
                aggEl.mBoundingBox, baseEl.mRefMap.(refEl.mNodesX) );

            for (j, aggNode) in enumerate( aggEl.mNodesInd ), (i, baseNode) in enumerate( baseEl.mNodesInd ) 
                push!(data, ( baseNode, aggNode, aggBasisFunVal[i,j] ) );
            end
        end
    end

    L = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        baseMesh.mNumNodes, aggMesh.mNumNodes );

    return L;
end

function aggdg_dg_interpolation2( aggMesh::Union{AgglomeratedDgMesh1, AgglomeratedDgMeshN}, 
    baseMesh::DgMesh )

    refEl = baseMesh.mRefEl;
    gaussQuadNodes = refEl.mGaussQuadNodes;
    gaussQuadWeights = refEl.mGaussQuadWeights;

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    for aggEl in aggMesh.mElements
        for baseElInd in aggEl.mBaseElementInds
            baseEl = baseMesh.mElements[ baseElInd ];
            temp = zeros( length(baseEl.mNodesInd), length(aggEl.mNodesInd) );
            aggBasisGQFunVal = evaluate_local_modal_basis_fun( aggMesh.mP, 
                aggEl.mBoundingBox, baseEl.mRefMap.(gaussQuadNodes) );

            for j = 1:length( aggEl.mNodesInd ), i = 1:length( baseEl.mNodesInd ), l = 1:length(gaussQuadNodes)
                temp[i,j] += baseEl.mJacobian * gaussQuadWeights[l] * 
                    refEl.mBasisGQFunVal[l,i] * aggBasisGQFunVal[l,j];
            end

            for (j, aggNode) in enumerate( aggEl.mNodesInd ), (i, baseNode) in enumerate( baseEl.mNodesInd ) 
                push!(data, ( baseNode, aggNode, temp[i,j] ) );
            end
        end
    end

    N = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        baseMesh.mNumNodes, aggMesh.mNumNodes );

    return sp.sparse( baseMesh.mMassMatrixLU \ Array(N) );
end

############################################################################################
# interpolation between AgglomeratedDgMesh and CgMesh
############################################################################################

function aggdg_cg_interpolation( aggMesh::Union{AgglomeratedDgMesh1, AgglomeratedDgMeshN}, 
    baseMesh::CgMesh, mesh::Mesh, interpFlag::Int64 )

    if interpFlag == 0 || interpFlag == 1
        data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

        refEl = baseMesh.mRefEl;
        gaussQuadNodes = refEl.mGaussQuadNodes;
        gaussQuadWeights = refEl.mGaussQuadWeights;

        for aggEl in aggMesh.mElements
            for baseElInd in aggEl.mBaseElementInds
                baseEl = baseMesh.mElements[ baseElInd ];
                temp = zeros( length(baseEl.mNodesInd), length(aggEl.mNodesInd) );
                aggBasisGQFunVal = evaluate_local_modal_basis_fun( aggMesh.mP, 
                    aggEl.mBoundingBox, baseEl.mRefMap.(gaussQuadNodes) );
    
                for j = 1:length( aggEl.mNodesInd ), i = 1:length( baseEl.mNodesInd ), l = 1:length(gaussQuadNodes)
                    temp[i,j] += baseEl.mJacobian * gaussQuadWeights[l] * 
                        refEl.mBasisGQFunVal[l,i] * aggBasisGQFunVal[l,j];
                end
    
                for (j, aggNode) in enumerate( aggEl.mNodesInd ), (i, baseNode) in enumerate( baseEl.mNodesInd ) 
                    push!(data, ( baseNode, aggNode, temp[i,j] ) );
                end
            end
        end
    
        N = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
            baseMesh.mNumNodes, aggMesh.mNumNodes );
    elseif interpFlag == 2
        data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

        refEl = baseMesh.mRefEl;

        for aggEl in aggMesh.mElements
            for baseElInd in aggEl.mBaseElementInds
                baseEl = baseMesh.mElements[ baseElInd ];
                aggBasisFunVal = evaluate_local_modal_basis_fun( aggMesh.mP, 
                    aggEl.mBoundingBox, baseEl.mRefMap.(refEl.mNodesX) );

                for j = 1:length( aggEl.mNodesInd )
                    aggNode = aggEl.mNodesInd[j];
                    
                    for i = 1:2
                        baseNode = baseEl.mNodesInd[i];
                        vert = mesh.mVertices[baseNode];
                        if isBoundary( vert )
                            push!( data, (baseNode, aggNode, aggBasisFunVal[i,j] ) );
                        else
                            push!( data, (baseNode, aggNode, 0.5*aggBasisFunVal[i,j] ) );
                        end
                    end

                    for i = 3:length( baseEl.mNodesInd )
                        baseNode = baseEl.mNodesInd[i];
                        push!( data, ( baseNode, aggNode, aggBasisFunVal[i,j] ) );
                    end
                end
            end
        end

        L = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
            baseMesh.mNumNodes, aggMesh.mNumNodes );
    else
        throw( ArgumentError( "Only implemented for interpFlag = 0, 1, or 2." ) );
    end

    if interpFlag == 0
        L = baseMesh.mMassMatrixLU \ Array(N);
    elseif interpFlag == 1
        lumpedMassMatrixVec = zeros( baseMesh.mNumNodes );
        for j = 1:baseMesh.mNumNodes
            lumpedMassMatrixVec[j] = sum( baseMesh.mMassMatrix[j,:] );
        end

        L = la.Diagonal( lumpedMassMatrixVec ) \ N;
    end

    return L;
end