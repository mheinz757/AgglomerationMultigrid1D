struct AgglomeratedDgVertex
    mIndex::Int64
    mX::Float64
    mFaces::Vector{Int64}

    AgglomeratedDgVertex( mIndex, mX, mFaces = zeros(Int64, 2) ) = new( mIndex, mX, mFaces );
end

struct AgglomeratedDgElement1 <: AbstractAgglomeratedDgElement
    mIndex::Int64
    mP::Int64
    mNodesInd::Vector{Int64}

    mSubAggElementInds::Vector{Int64}
    mBaseElementInds::Vector{Int64} # same as mSubElementInds

    mBoundingBox::Vector{Float64} # gives coordinates of the bounding box
    mBasisGQFunVal::Vector{Matrix{Float64}}
    mBasisDerivVal::Vector{Float64}

    mVertices::Vector{AgglomeratedDgVertex} # could maybe just be a vector of Integers
    mVertices2::Vector{Vertex}
    mBdBasisGQFunVal::Vector{Vector{Float64}} # maybe store as a multidimensional array
                                              # also will be Vector{Matrix{Float64}} in 2d

    mNeighborVertInd::Vector{Int64} # for each edge associated with this agglomerated 
                                    # element, stores the local index of the edge in the 
                                    # neighboring element that shares the edge
end

struct AgglomeratedDgMesh1 <: AbstractMesh
    mP::Int64
    mElements::Vector{AgglomeratedDgElement1}
    mNumNodes::Int64

    mAllVertices::Vector{AgglomeratedDgVertex}
    mVertices::Vector{AgglomeratedDgVertex}
    mSwitch::Vector{Int64} # switch function for each vertex (edge in 2d)

    mMassMatrix::sp.SparseMatrixCSC{Float64, Int64} # can switch to bd.BlockDiagonal{Float64, Matrix{Float64}}
    mMassMatrixLU::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}

    mGaussQuadNodes::Vector{Float64}
    mGaussQuadWeights::Vector{Float64}

    mBaseToAgglomeratedInd::Vector{Int64} # stores which agglomerated element each base element got put into
end

struct AgglomeratedDgElementN <: AbstractAgglomeratedDgElement
    mIndex::Int64
    mP::Int64
    mNodesInd::Vector{Int64}

    mSubAggElementInds::Vector{Int64}
    mBaseElementInds::Vector{Int64}

    mBoundingBox::Vector{Float64} # gives coordinates of the bounding box
    mBasisGQFunVal::Vector{Matrix{Float64}}
    mBasisDerivVal::Vector{Float64}
end

struct AgglomeratedDgMeshN <: AbstractMesh
    mP::Int64
    mElements::Vector{AgglomeratedDgElementN}
    mNumNodes::Int64

    mMassMatrix::sp.SparseMatrixCSC{Float64, Int64} # can switch to bd.BlockDiagonal{Float64, Matrix{Float64}}
    mMassMatrixLU::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}

    mGaussQuadNodes::Vector{Float64}
    mGaussQuadWeights::Vector{Float64}
end

############################################################################################
# AgglomeratedDgVertex utility
############################################################################################

function isBoundary( vertex::AgglomeratedDgVertex )
    return vertex.mFaces[2] < 1;
end

############################################################################################
# AgglomeratedDgElement1 creation and utility
############################################################################################

function AgglomeratedDgElement1( mIndex, mP, mBaseElementInds, 
    baseMesh::Union{CgMesh, DgMesh}, mesh::Mesh, allVertices::Vector{AgglomeratedDgVertex} )

    mNodesInd = ((mIndex - 1) * (mP + 1) + 1):(mIndex * (mP + 1));
    mSubAggElementInds = mBaseElementInds;

    # find the bounding "box"
    min_x = Inf;
    max_x = -Inf;
    for elInd in mBaseElementInds
        min_x = min( min_x, baseMesh.mElements[elInd].mNodesX[1] );
        max_x = max( max_x, baseMesh.mElements[elInd].mNodesX[2] );
    end
    mBoundingBox = [min_x, max_x];
    # h = max_x - min_x;
    # xc = ( min_x + max_x ) / 2.0;

    # find the basis GQ Fun Val
    # local modal basis, phi1 = 1, phi2 = 2*(x - xc)/h
    gaussQuadNodes, = gauss_quad( 2*mP );
    mBasisGQFunVal = Vector{ Matrix{Float64} }( undef, length(mBaseElementInds) );
    for (i, elInd) in enumerate( mBaseElementInds )
        el = baseMesh.mElements[elInd];
        elGaussQuadNodes = el.mRefMap.(gaussQuadNodes);
        mBasisGQFunVal[i] = evaluate_local_modal_basis_fun( mP, mBoundingBox, 
            elGaussQuadNodes );

        # mBasisGQFunVal[i] = zeros( length( elGaussQuadNodes ), mP + 1 );
        # if mP == 0
        #     mBasisGQFunVal[i][:,1] = ones( length( elGaussQuadNodes ) );
        # elseif mP == 1
        #     mBasisGQFunVal[i][:,1] = ones( length( elGaussQuadNodes ) );
        #     for (l, node) in enumerate( elGaussQuadNodes )
        #         mBasisGQFunVal[i][l,2] = 2 * ( node - xc ) / h;
        #     end
        #     # for (l, node) in enumerate( elGaussQuadNodes )
        #     #     mBasisGQFunVal[i][l,1] = 0.5 - ( node - xc ) / h
        #     #     mBasisGQFunVal[i][l,2] = 0.5 + ( node - xc ) / h;
        #     # end
        # else
        #     throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
        # end
    end

    # find the basis derivative value (always a constant)
    mBasisDerivVal = evaluate_local_modal_basis_deriv( mP, mBoundingBox );
    # if mP == 0
    #     mBasisDerivVal = [0.0];
    # elseif mP == 1
    #     mBasisDerivVal = [0.0, 2.0 / h];
    #     # mBasisDerivVal = [ -1.0 / h, 1.0 / h ];
    # else
    #     throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
    # end

    # find the boundary vertices (might be better implementation of this)
    mVertices = Vector{AgglomeratedDgVertex}(undef, 0);
    mVertices2 = Vector{Vertex}(undef, 0);
    for i in mBaseElementInds
        el = mesh.mFaces[i];
        for vert in el.mVertices
            isBd = !((vert.mFaces[1] in mBaseElementInds) && 
                (vert.mFaces[2] in mBaseElementInds));

            if isBd
                push!( mVertices2, vert );
                push!( mVertices, allVertices[vert.mIndex] );
            end
        end
    end

    # get the values of the basis functions at each boundary
    mBdBasisGQFunVal = Vector{ Vector{Float64} }( undef, length(mVertices) );
    for (i, vert) in enumerate( mVertices )
        x = vert.mX;
        mBdBasisGQFunVal[i] = evaluate_local_modal_basis_fun( mP, mBoundingBox, x )[1,:];
        # mBdBasisGQFunVal[i] = zeros( mP+1 );
        # if mP == 0
        #     mBdBasisGQFunVal[i][1] = 1.0;
        # elseif mP == 1
        #     mBdBasisGQFunVal[i][1] = 1.0;
        #     mBdBasisGQFunVal[i][2] = 2 * ( x - xc ) / h;
        #     # mBdBasisGQFunVal[i][1] = 0.5 - ( x - xc ) / h;
        #     # mBdBasisGQFunVal[i][2] = 0.5 + ( x - xc ) / h;
        # else
        #     throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
        # end
    end

    # mNeighborVertInd still unused
    mNeighborVertInd = Vector{Int64}( undef, 0 );

    return AgglomeratedDgElement1( mIndex, mP, mNodesInd, mSubAggElementInds, 
        mBaseElementInds, mBoundingBox, mBasisGQFunVal, mBasisDerivVal, mVertices, 
        mVertices2, mBdBasisGQFunVal, mNeighborVertInd );
end

function AgglomeratedDgElement1( mIndex, mP, mBaseElementInds, 
    baseMesh::Union{CgMesh, DgMesh}, mesh::Mesh, allVertices::Vector{AgglomeratedDgVertex}, 
    gaussQuadNodes::Vector{Float64} )

    mNodesInd = ((mIndex - 1) * (mP + 1) + 1):(mIndex * (mP + 1));
    mSubAggElementInds = mBaseElementInds;

    # find the bounding "box" (assumes the baseMesh has at least p >= 1)
    min_x = Inf;
    max_x = -Inf;
    for elInd in mBaseElementInds
        min_x = min( min_x, baseMesh.mElements[elInd].mNodesX[1] );
        max_x = max( max_x, baseMesh.mElements[elInd].mNodesX[2] );
    end
    mBoundingBox = [min_x, max_x];
    # h = max_x - min_x;
    # xc = ( min_x + max_x ) / 2.0;

    # find the basis GQ Fun Val
    # local modal basis, phi1 = 1, phi2 = 2*(x - xc)/h
    mBasisGQFunVal = Vector{ Matrix{Float64} }( undef, length(mBaseElementInds) );
    for (i, elInd) in enumerate( mBaseElementInds )
        el = baseMesh.mElements[elInd];
        elGaussQuadNodes = el.mRefMap.(gaussQuadNodes);
        mBasisGQFunVal[i] = evaluate_local_modal_basis_fun( mP, mBoundingBox, 
            elGaussQuadNodes );

        # mBasisGQFunVal[i] = zeros( length( elGaussQuadNodes ), mP + 1 );
        # if mP == 0
        #     mBasisGQFunVal[i][:,1] = ones( length( elGaussQuadNodes ) );
        # elseif mP == 1
        #     mBasisGQFunVal[i][:,1] = ones( length( elGaussQuadNodes ) );
        #     for (l, node) in enumerate( elGaussQuadNodes )
        #         mBasisGQFunVal[i][l,2] = 2 * ( node - xc ) / h;
        #     end
        #     # for (l, node) in enumerate( elGaussQuadNodes )
        #     #     mBasisGQFunVal[i][l,1] = 0.5 - ( node - xc ) / h
        #     #     mBasisGQFunVal[i][l,2] = 0.5 + ( node - xc ) / h;
        #     # end
        # else
        #     throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
        # end
    end

    # find the basis derivative value (always a constant)
    mBasisDerivVal = evaluate_local_modal_basis_deriv( mP, mBoundingBox );
    # if mP == 0
    #     mBasisDerivVal = [0.0];
    # elseif mP == 1
    #     mBasisDerivVal = [0.0, 2.0 / h];
    #     # mBasisDerivVal = [ -1.0 / h, 1.0 / h ];
    # else
    #     throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
    # end

    # find the boundary vertices (might be better implementation of this)
    mVertices = Vector{AgglomeratedDgVertex}(undef, 0);
    mVertices2 = Vector{Vertex}(undef, 0);
    for i in mBaseElementInds
        el = mesh.mFaces[i];
        for vert in el.mVertices
            isBd = !((vert.mFaces[1] in mBaseElementInds) && 
                (vert.mFaces[2] in mBaseElementInds));

            if isBd
                push!( mVertices2, vert );
                push!( mVertices, allVertices[vert.mIndex] );
            end
        end
    end

    # get the values of the basis functions at each boundary
    mBdBasisGQFunVal = Vector{ Vector{Float64} }( undef, length(mVertices) );
    for (i, vert) in enumerate( mVertices )
        x = vert.mX;
        mBdBasisGQFunVal[i] = evaluate_local_modal_basis_fun( mP, mBoundingBox, x )[1,:];
        # mBdBasisGQFunVal[i] = zeros( mP+1 );
        # if mP == 0
        #     mBdBasisGQFunVal[i][1] = 1.0;
        # elseif mP == 1
        #     mBdBasisGQFunVal[i][1] = 1.0;
        #     mBdBasisGQFunVal[i][2] = 2 * ( x - xc ) / h;
        #     # mBdBasisGQFunVal[i][1] = 0.5 - ( x - xc ) / h;
        #     # mBdBasisGQFunVal[i][2] = 0.5 + ( x - xc ) / h;
        # else
        #     throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
        # end
    end

    # mNeighborVertInd still unused
    mNeighborVertInd = Vector{Int64}( undef, 0 );

    return AgglomeratedDgElement1( mIndex, mP, mNodesInd, mSubAggElementInds, 
        mBaseElementInds, mBoundingBox, mBasisGQFunVal, mBasisDerivVal, mVertices, 
        mVertices2, mBdBasisGQFunVal, mNeighborVertInd );
end

function resolve_vertices!( elements::Vector{AgglomeratedDgElement1} )

    for el in elements
        for vert in el.mVertices
            if vert.mFaces[1] == 0
                vert.mFaces[1] = el.mIndex;
            elseif vert.mFaces[2] == 0
                vert.mFaces[2] = el.mIndex;
            else
                error( "Vertex can only neighbor two elements." );
            end
        end
    end

    return
end

function evaluate_local_modal_basis_fun( p, boundingBox, nodes )

    basisFunVal = zeros( length(nodes), p+1 );

    if p == 0
        basisFunVal[:,1] = ones( length(nodes) );
        return basisFunVal;
    elseif p == 1
        xC = ( boundingBox[1] + boundingBox[2] ) / 2.0;
        h = boundingBox[2] - boundingBox[1];

        basisFunVal[:,1] = ones( length(nodes) );
        basisFunVal[:,2] .= 2 * ( nodes .- xC ) / h;
        return basisFunVal;
    else
        throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
        return
    end
end

function evaluate_local_modal_basis_deriv( p, boundingBox )
    if p == 0
        return [0.0]
    elseif p == 1
        h = boundingBox[2] - boundingBox[1];
        return [0.0, 2.0 / h];
    else
        throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
        return
    end
end

############################################################################################
# AgglomeratedDgMesh1 creation
############################################################################################

function AgglomeratedDgMesh1( mP, mElements, mAllVertices, mVertices, 
    baseMesh::Union{CgMesh, DgMesh} )

    mNumNodes = length( mElements ) * (mP + 1);
    mGaussQuadNodes, mGaussQuadWeights = gauss_quad( 2*mP );

    # get mass matrix
    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    for el in mElements
        temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
        for ( k, baseElInd ) in enumerate( el.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( mGaussQuadNodes )
                temp[i,j] += baseEl.mJacobian * mGaussQuadWeights[l] * 
                    el.mBasisGQFunVal[k][l,i] * el.mBasisGQFunVal[k][l,j];
            end
        end

        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
            push!(data, ( node1, node2, temp[i,j] ) );
        end
    end

    mMassMatrix = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        mNumNodes, mNumNodes );
    mMassMatrixLU = la.lu(mMassMatrix);

    # get switch functions
    mSwitch = Vector{Int64}( undef, length( mVertices ) );

    for ( i, vert ) in enumerate( mVertices )
        if isBoundary( vert )
            # do something if the vertex is a boundary vertex in case of periodic b.c.
            el = mElements[ vert.mFaces[1] ];

            # if the boundary vertex is on the right, select its own face
            # otherwise, select *other* face
            if vert.mX > min(el.mVertices[1].mX, el.mVertices[2].mX)
                mSwitch[i] = 1;
            else
                mSwitch[i] = 2;
            end
        else
            el1 = mElements[ vert.mFaces[1] ];
            el2 = mElements[ vert.mFaces[1] ];

            x1 = max(el1.mVertices[1].mX, el1.mVertices[2].mX);
            x2 = max(el2.mVertices[1].mX, el2.mVertices[2].mX);

            # select index of face that is on the left of the vertex
            if x1 > x2
                mSwitch[i] = 2; 
            else
                mSwitch[i] = 1;
            end
        end
    end

    # mBaseToAgglomeratedInd still unused
    mBaseToAgglomeratedInd = Vector{Int64}( undef, 0 );

    return AgglomeratedDgMesh1( mP, mElements, mNumNodes, mAllVertices, mVertices, mSwitch, 
        mMassMatrix, mMassMatrixLU, mGaussQuadNodes, mGaussQuadWeights, 
        mBaseToAgglomeratedInd );
end

function AgglomeratedDgMesh1( mP, agg::Vector{Vector{Int64}}, mesh::Mesh, 
    baseMesh::Union{CgMesh, DgMesh} )

    # get gaussQuadNodes and weights for this order element
    mGaussQuadNodes, mGaussQuadWeights = gauss_quad( 2*mP );

    # create vectors for all potential agglomerated vertices and all the elements
    mAllVertices = Vector{AgglomeratedDgVertex}(undef, length( mesh.mVertices ) );
    for (i, vert) in enumerate( mesh.mVertices )
        mAllVertices[i] = AgglomeratedDgVertex( vert.mIndex, vert.mX )
    end

    # create all elements
    mElements = Vector{AgglomeratedDgElement1}(undef, length( agg ) );
    mNumNodes = length( mElements ) * (mP + 1);

    for (k, baseElInds) in enumerate( agg )
        mElements[k] = AgglomeratedDgElement1( k, mP, baseElInds, baseMesh, mesh, 
            mAllVertices, mGaussQuadNodes );
    end

    # find agglomerated vertices that are actually in the mesh, and update vertex.mFaces
    mVertices = Vector{AgglomeratedDgVertex}(undef, 0);
    for el in mElements
        for vert in el.mVertices
            if vert.mFaces[1] == 0
                vert.mFaces[1] = el.mIndex;
                push!( mVertices, vert );
            elseif vert.mFaces[2] == 0
                vert.mFaces[2] = el.mIndex;
            else
                error( "Vertex can only neighbor two elements." );
            end
        end
    end

    # get mass matrix
    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    for el in mElements
        temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
        for ( k, baseElInd ) in enumerate( el.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( mGaussQuadNodes )
                temp[i,j] += baseEl.mJacobian * mGaussQuadWeights[l] * 
                    el.mBasisGQFunVal[k][l,i] * el.mBasisGQFunVal[k][l,j];
            end
        end

        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
            push!(data, ( node1, node2, temp[i,j] ) );
        end
    end

    mMassMatrix = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        mNumNodes, mNumNodes );
    mMassMatrixLU = la.lu(mMassMatrix);

    # get switch functions
    mSwitch = Vector{Int64}( undef, length( mVertices ) );

    for ( i, vert ) in enumerate( mVertices )
        if isBoundary( vert )
            # do something if the vertex is a boundary vertex in case of periodic b.c.
            el = mElements[ vert.mFaces[1] ];

            # if the boundary vertex is on the right, select its own face
            # otherwise, select *other* face
            if vert.mX > min(el.mVertices[1].mX, el.mVertices[2].mX)
                mSwitch[i] = 1;
            else
                mSwitch[i] = 2;
            end
        else
            el1 = mElements[ vert.mFaces[1] ];
            el2 = mElements[ vert.mFaces[1] ];

            x1 = max(el1.mVertices[1].mX, el1.mVertices[2].mX);
            x2 = max(el2.mVertices[1].mX, el2.mVertices[2].mX);

            # select index of face that is on the left of the vertex
            if x1 > x2
                mSwitch[i] = 2; 
            else
                mSwitch[i] = 1;
            end
        end
    end

    # mBaseToAgglomeratedInd still unused
    mBaseToAgglomeratedInd = Vector{Int64}( undef, 0 );

    return AgglomeratedDgMesh1( mP, mElements, mNumNodes, mAllVertices, mVertices, mSwitch, 
        mMassMatrix, mMassMatrixLU, mGaussQuadNodes, mGaussQuadWeights, 
        mBaseToAgglomeratedInd );
end

############################################################################################
# AgglomeratedDgElementN creation
############################################################################################

function AgglomeratedDgElementN( mIndex, mP, mSubAggElementInds, 
    subAggMesh::Union{AgglomeratedDgMesh1, AgglomeratedDgMeshN}, 
    baseMesh::Union{CgMesh, DgMesh} )

    mNodesInd = ((mIndex - 1) * (mP + 1) + 1):(mIndex * (mP + 1));

    # find the base element indices for this element
    mBaseElementInds = Vector{Int64}(undef, 0);
    for elInd in mSubAggElementInds
        append!( mBaseElementInds, subAggMesh.mElements[elInd].mBaseElementInds );
    end

    # find the bounding "box"
    min_x = Inf;
    max_x = -Inf;
    for elInd in mSubAggElementInds
        min_x = min( min_x, subAggMesh.mElements[elInd].mBoundingBox[1] );
        max_x = max( max_x, subAggMesh.mElements[elInd].mBoundingBox[2] );
    end
    mBoundingBox = [min_x, max_x];
    # h = max_x - min_x;
    # xc = ( min_x + max_x ) / 2.0;

    # find the basis GQ Fun Val
    # local modal basis, phi1 = 1, phi2 = 2*(x - xc)/h
    gaussQuadNodes, = gauss_quad( 2*mP );
    mBasisGQFunVal = Vector{ Matrix{Float64} }( undef, length(mBaseElementInds) );
    for (i, elInd) in enumerate( mBaseElementInds )
        el = baseMesh.mElements[elInd];
        elGaussQuadNodes = el.mRefMap.(gaussQuadNodes);
        mBasisGQFunVal[i] = evaluate_local_modal_basis_fun( mP, mBoundingBox, 
            elGaussQuadNodes );

        # mBasisGQFunVal[i] = zeros( length( elGaussQuadNodes ), mP + 1 );
        # if mP == 0
        #     mBasisGQFunVal[i][:,1] = ones( length(elGaussQuadNodes ) );
        # elseif mP == 1
        #     mBasisGQFunVal[i][:,1] = ones( length(elGaussQuadNodes ) );
        #     for (l, node) in enumerate( elGaussQuadNodes )
        #         mBasisGQFunVal[i][l,2] = 2 * ( node - xc ) / h;
        #     end
        # else
        #     throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
        # end
    end

    # find the basis derivative value (always a constant)
    mBasisDerivVal = evaluate_local_modal_basis_deriv( mP, mBoundingBox );
    # if mP == 0
    #     mBasisDerivVal = [0.0];
    # elseif mP == 1
    #     mBasisDerivVal = [0.0, 2.0 / h];
    # else
    #     throw( ArgumentError( "Only implemented for p = 0 and p = 1." ) );
    # end

    return AgglomeratedDgElementN( mIndex, mP, mNodesInd, mSubAggElementInds, 
        mBaseElementInds, mBoundingBox, mBasisGQFunVal, mBasisDerivVal );
end

############################################################################################
# AgglomeratedDgMeshN creation
############################################################################################

function AgglomeratedDgMeshN( mP, mElements, baseMesh::Union{CgMesh, DgMesh} )

    mNumNodes = length( mElements ) * (mP + 1);
    mGaussQuadNodes, mGaussQuadWeights = gauss_quad( 2*mP );

    # get mass matrix
    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    for el in mElements
        temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
        for ( k, baseElInd ) in enumerate( el.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( mGaussQuadNodes )
                temp[i,j] += baseEl.mJacobian * mGaussQuadWeights[l] * 
                    el.mBasisGQFunVal[k][l,i] * el.mBasisGQFunVal[k][l,j];
            end
        end

        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
            push!(data, ( node1, node2, temp[i,j] ) );
        end
    end

    mMassMatrix = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        mNumNodes, mNumNodes );
    mMassMatrixLU = la.lu(mMassMatrix);

    return AgglomeratedDgMeshN( mP, mElements, mNumNodes, mMassMatrix, mMassMatrixLU, 
        mGaussQuadNodes, mGaussQuadWeights );
end

function AgglomeratedDgMeshN( mP, agg::Vector{Vector{Int64}}, 
    subAggMesh::Union{AgglomeratedDgMesh1, AgglomeratedDgMeshN}, 
    baseMesh::Union{CgMesh, DgMesh} )

    # get gaussQuadNodes and weights for this order element
    mGaussQuadNodes, mGaussQuadWeights = gauss_quad( 2*mP );

    # create all elements
    mElements = Vector{AgglomeratedDgElementN}(undef, length( agg ) );
    mNumNodes = length( mElements ) * (mP + 1);

    for (k, subAggElInds) in enumerate( agg )
        mElements[k] = AgglomeratedDgElementN( k, mP, subAggElInds, subAggMesh, baseMesh );
    end

    # get mass matrix
    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    for el in mElements
        temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
        for ( k, baseElInd ) in enumerate( el.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( mGaussQuadNodes )
                temp[i,j] += baseEl.mJacobian * mGaussQuadWeights[l] * 
                    el.mBasisGQFunVal[k][l,i] * el.mBasisGQFunVal[k][l,j];
            end
        end

        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
            push!(data, ( node1, node2, temp[i,j] ) );
        end
    end

    mMassMatrix = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        mNumNodes, mNumNodes );
    mMassMatrixLU = la.lu(mMassMatrix);

    return AgglomeratedDgMeshN( mP, mElements, mNumNodes, mMassMatrix, mMassMatrixLU, 
        mGaussQuadNodes, mGaussQuadWeights );
end

############################################################################################
# operators and rhs for AgglomeratedDgMesh1
############################################################################################

function dg_flux_operators( aggDgMesh::AgglomeratedDgMesh1, baseMesh::Union{CgMesh, DgMesh}, 
    bdCond::BoundaryCondition, CDir::Float64 )

    dataG = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    dataD = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    dataC = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    if aggDgMesh.mP >= 1
        # do volume integrals for gradient and divergence
        for el in aggDgMesh.mElements
            temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
            for (k, baseElInd) in enumerate( el.mBaseElementInds )
                baseEl = baseMesh.mElements[ baseElInd ];
                for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( aggDgMesh.mGaussQuadNodes )
                    temp[i,j] += baseEl.mJacobian * aggDgMesh.mGaussQuadWeights[l] * 
                        el.mBasisDerivVal[i] * el.mBasisGQFunVal[k][l,j];
                end
            end

            for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                push!(dataG, ( node1, node2, temp[i,j] ) );
                push!(dataD, ( node1, node2, temp[i,j] ) );
            end
        end

        # do edge integrals
        for ( i, vert ) in enumerate( aggDgMesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # Gradient: do nothing because uhat = g_D
                    # Divergence: use qhat = q_h +/- CDir(u_h - g_D)
                    ### only contribution here is qhat = q_h
                    # C Matrix: use qhat = q_h +/- CDir(u_h - g_D)
                    ### contribution here is qhat = +/- CDir u_h

                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        sign = -1.0;
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataD, ( node1, node2, -sign * el.mBdBasisGQFunVal[1][i] *
                                el.mBdBasisGQFunVal[1][j] ) );
                            push!(dataC, ( node1, node2, CDir * el.mBdBasisGQFunVal[1][i] *
                                el.mBdBasisGQFunVal[1][j] ) );
                        end
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        sign = 1.0;
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataD, ( node1, node2, -sign * el.mBdBasisGQFunVal[2][i] *
                                el.mBdBasisGQFunVal[2][j] ) );
                            push!(dataC, ( node1, node2, CDir * el.mBdBasisGQFunVal[2][i] *
                                el.mBdBasisGQFunVal[2][j] ) );
                        end
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # Gradient: uhat = u_h, u on the element itself
                    # Divergence: do nothing because qhat = g_N
                    # C_matrix: do nothing

                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        sign = -1.0;
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataG, ( node1, node2, -sign * el.mBdBasisGQFunVal[1][i] *
                                el.mBdBasisGQFunVal[1][j] ) );
                        end
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        sign = 1.0;
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataG, ( node1, node2, -sign * el.mBdBasisGQFunVal[2][i] *
                                el.mBdBasisGQFunVal[2][j] ) );
                        end
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                else
                    error( "Boundary vertex is not included in the boundary condition." );
                end
            else
                # first, get elements which correspond to uhat and qhat for this vertex 
                # uhat = u_L, qhat = q_R
                S = aggDgMesh.mSwitch[i];
                uhatEl = aggDgMesh.mElements[ vert.mFaces[S] ];
                qhatEl = aggDgMesh.mElements[ vert.mFaces[S%2 + 1] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    el = aggDgMesh.mElements[k];
                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element 
                        # second vertex for uhatEl, first vertex for qhatEl
                        # first vertex for el
                        sign = -1.0;
                        for (j, node2) in enumerate( uhatEl.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataG, ( node1, node2, -sign * el.mBdBasisGQFunVal[1][i] *
                                uhatEl.mBdBasisGQFunVal[2][j] ) );
                        end
                        for (j, node2) in enumerate( qhatEl.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataD, ( node1, node2, -sign * el.mBdBasisGQFunVal[1][i] *
                                qhatEl.mBdBasisGQFunVal[1][j] ) );
                        end
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element 
                        # second vertex for uhatEl, first vertex for qhatEl
                        # second vertex for el
                        sign = 1.0;
                        for (j, node2) in enumerate( uhatEl.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataG, ( node1, node2, -sign * el.mBdBasisGQFunVal[2][i] *
                                uhatEl.mBdBasisGQFunVal[2][j] ) );
                        end
                        for (j, node2) in enumerate( qhatEl.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataD, ( node1, node2, -sign * el.mBdBasisGQFunVal[2][i] *
                                qhatEl.mBdBasisGQFunVal[1][j] ) );
                        end
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    else
        # no volume integrals because derivative is always zero

        # do edge integrals
        for ( i, vert ) in enumerate( aggDgMesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # Gradient: do nothing because uhat = g_D
                    # Divergence: use qhat = q_h +/- CDir(u_h - g_D)
                    ### only contribution here is qhat = q_h
                    # C Matrix: use qhat = q_h +/- CDir(u_h - g_D)
                    ### contribution here is qhat = +/- CDir u_h

                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        sign = -1.0;
                        push!( dataD, ( el.mNodesInd[1], el.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[1][1] * el.mBdBasisGQFunVal[1][1] ) );
                        push!( dataC, ( el.mNodesInd[1], el.mNodesInd[1], CDir * 
                            el.mBdBasisGQFunVal[1][1] * el.mBdBasisGQFunVal[1][1] ) );
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        sign = 1.0;
                        push!( dataD, ( el.mNodesInd[1], el.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[2][1] * el.mBdBasisGQFunVal[2][1] ) );
                        push!( dataC, ( el.mNodesInd[1], el.mNodesInd[1], CDir * 
                            el.mBdBasisGQFunVal[2][1] * el.mBdBasisGQFunVal[2][1] ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # Gradient: uhat = u_h, u on the element itself
                    # Divergence: do nothing because qhat = g_N
                    # C_matrix: do nothing

                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        sign = -1.0;
                        push!( dataG, ( el.mNodesInd[1], el.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[1][1] * el.mBdBasisGQFunVal[1][1] ) );
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        sign = 1.0;
                        push!( dataG, ( el.mNodesInd[1], el.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[2][1] * el.mBdBasisGQFunVal[2][1] ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                else
                    error( "Boundary vertex is not included in the boundary condition." )
                end
            else
                # first, get element which corresponds to uhat and qhat for this vertex 
                # uhat = u_L, qhat = q_R
                S = aggDgMesh.mSwitch[i];
                uhatEl = aggDgMesh.mElements[ vert.mFaces[S] ];
                qhatEl = aggDgMesh.mElements[ vert.mFaces[S%2 + 1] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    el = aggDgMesh.mElements[k];
                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element 
                        # second vertex for uhatEl, first vertex for qhatEl
                        # first vertex for el
                        sign = -1.0;
                        push!( dataG, ( el.mNodesInd[1], uhatEl.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[1][1] * uhatEl.mBdBasisGQFunVal[2][1] ) );
                        push!( dataD, ( el.mNodesInd[1], qhatEl.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[1][1] * qhatEl.mBdBasisGQFunVal[1][1] ) );
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element 
                        # second vertex for uhatEl, first vertex for qhatEl
                        # second vertex for el
                        sign = 1.0;
                        push!( dataG, ( el.mNodesInd[1], uhatEl.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[2][1] * uhatEl.mBdBasisGQFunVal[2][1] ) );
                        push!( dataD, ( el.mNodesInd[1], qhatEl.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[2][1] * qhatEl.mBdBasisGQFunVal[1][1] ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    end

    G = sp.sparse( (x->x[1]).(dataG), (x->x[2]).(dataG), (x->x[3]).(dataG), 
        aggDgMesh.mNumNodes, aggDgMesh.mNumNodes );
    D = sp.sparse( (x->x[1]).(dataD), (x->x[2]).(dataD), (x->x[3]).(dataD), 
        aggDgMesh.mNumNodes, aggDgMesh.mNumNodes );
    C = sp.sparse( (x->x[1]).(dataC), (x->x[2]).(dataC), (x->x[3]).(dataC), 
        aggDgMesh.mNumNodes, aggDgMesh.mNumNodes );

    return G, D, C;
end

function dg_flux_rhs( aggDgMesh::AgglomeratedDgMesh1, baseMesh::Union{CgMesh, DgMesh}, 
    func::Function, bdCond::BoundaryCondition, CDir::Float64 )

    f = zeros( aggDgMesh.mNumNodes );
    r = zeros( aggDgMesh.mNumNodes );

    # do volume integrals
    for el in aggDgMesh.mElements
        for (k, baseElInd) in enumerate( el.mBaseElementInds )
            baseEl = baseMesh.mElements[ baseElInd ];
            for (i, node) in enumerate( el.mNodesInd ), l = 1:length( aggDgMesh.mGaussQuadNodes )
                f[node] += baseEl.mJacobian * aggDgMesh.mGaussQuadWeights[l] * 
                    el.mBasisGQFunVal[k][l,i] * 
                    func( baseEl.mRefMap( aggDgMesh.mGaussQuadNodes[l] ) );
            end
        end
    end

    if aggDgMesh.mP >= 1
        # do edge integrals for dirichlet boundaries
        for ( i, nodeIdx ) in enumerate( bdCond.mDirNodes )
            vert = aggDgMesh.mAllVertices[ nodeIdx ];
            dirVal = bdCond.mDirVals[i];

            el = aggDgMesh.mElements[ vert.mFaces[1] ];

            if vert == el.mVertices[1]
                # the vertex is the incoming boundary of this element
                sign = -1.0;
                for (i, node) in enumerate( el.mNodesInd )
                    f[ node ] += CDir * dirVal * el.mBdBasisGQFunVal[1][i];
                    r[ node ] += sign * dirVal * el.mBdBasisGQFunVal[1][i];
                end
            elseif vert == el.mVertices[2]
                # the vertex is the outgoing boundary of this element
                sign = 1.0;
                for (i, node) in enumerate( el.mNodesInd )
                    f[ node ] += CDir * dirVal * el.mBdBasisGQFunVal[2][i];
                    r[ node ] += sign * dirVal * el.mBdBasisGQFunVal[2][i];
                end
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end 

        # do edge integrals for neumann boundaries
        for nodeIdx in bdCond.mNeuNodes
            vert = aggDgMesh.mAllVertices[ nodeIdx ];

            el = aggDgMesh.mElements[ vert.mFaces[1] ];

            if vert == el.mVertices[1]
                # the vertex is the incoming boundary of this element
                neuVal = bdCond.mBdCond[1][2];
                sign = -1.0;
                for (i, node) in enumerate( el.mNodesInd )
                    f[ node ] += sign * neuVal * el.mBdBasisGQFunVal[1][i];
                end
            elseif vert == el.mVertices[2]
                # the vertex is the outgoing boundary of this element
                neuVal = bdCond.mBdCond[2][2];
                sign = 1.0;
                for (i, node) in enumerate( el.mNodesInd )
                    f[ node ] += sign * neuVal * el.mBdBasisGQFunVal[2][i];
                end
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end 
    else
        # do edge integrals for dirichlet boundaries
        for ( i, nodeIdx ) in enumerate( bdCond.mDirNodes )
            vert = aggDgMesh.mAllVertices[ nodeIdx ];
            dirVal = bdCond.mDirVals[i];

            el = aggDgMesh.mElements[ vert.mFaces[1] ];

            if vert == el.mVertices[1]
                # the vertex is the incoming boundary of this element
                sign = -1.0;
                f[ el.mNodesInd[1] ] += CDir * dirVal * el.mBdBasisGQFunVal[1][1];
                r[ el.mNodesInd[1] ] += sign * dirVal * el.mBdBasisGQFunVal[1][1];
            elseif vert == el.mVertices[2]
                # the vertex is the outgoing boundary of this element
                sign = 1.0;
                f[ el.mNodesInd[1] ] += CDir * dirVal * el.mBdBasisGQFunVal[2][1];
                r[ el.mNodesInd[1] ] += sign * dirVal * el.mBdBasisGQFunVal[2][1];
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end 

        # do edge integrals for neumann boundaries
        for nodeIdx in bdCond.mNeuNodes
            vert = aggDgMesh.mAllVertices[ nodeIdx ];

            el = aggDgMesh.mElements[ vert.mFaces[1] ];

            if vert == el.mVertices[1]
                # the vertex is the incoming boundary of this element
                neuVal = bdCond.mBdCond[1][2];
                sign = -1.0;
                f[ el.mNodesInd[1] ] += sign * neuVal * el.mBdBasisGQFunVal[1][1];
            elseif vert == el.mVertices[2]
                # the vertex is the outgoing boundary of this element
                neuVal = bdCond.mBdCond[2][2];
                sign = 1.0;
                f[ el.mNodesInd[1] ] += sign * neuVal * el.mBdBasisGQFunVal[2][1];
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end
    end

    return f, r;
end

############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

function gradient( aggDgMesh::AgglomeratedDgMesh1, baseMesh::Union{CgMesh, DgMesh}, 
    bdCond::BoundaryCondition )

    dataG = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    if aggDgMesh.mP >= 1
        # do volume integrals for gradient
        for el in aggDgMesh.mElements
            temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
            for (k, baseElInd) in enumerate( el.mBaseElementInds )
                baseEl = baseMesh.mElements[ baseElInd ];
                for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( aggDgMesh.mGaussQuadNodes )
                    temp[i,j] += baseEl.mJacobian * aggDgMesh.mGaussQuadWeights[l] * 
                        el.mBasisDerivVal[i] * el.mBasisGQFunVal[k][l,j];
                end
            end

            for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                push!(dataG, ( node1, node2, temp[i,j] ) );
            end
        end

        # do edge integrals
        for ( i, vert ) in enumerate( aggDgMesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # do nothing because uhat = g_D
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # uhat = u_h, u on the element itself
                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        sign = -1.0;
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataG, ( node1, node2, -sign * el.mBdBasisGQFunVal[1][i] *
                                el.mBdBasisGQFunVal[1][j] ) );
                        end
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        sign = 1.0;
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataG, ( node1, node2, -sign * el.mBdBasisGQFunVal[2][i] *
                                el.mBdBasisGQFunVal[2][j] ) );
                        end
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                else
                    error( "Boundary vertex is not included in the boundary condition." );
                end
            else
                # first, get element which corresponds to uhat for this vertex (uhat = u_L)
                S = aggDgMesh.mSwitch[i];
                uhatEl = aggDgMesh.mElements[ vert.mFaces[S] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    el = aggDgMesh.mElements[k];
                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element (so second 
                        # vertex for uhatEl, but first vertex for el)
                        sign = -1.0;
                        for (j, node2) in enumerate( uhatEl.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataG, ( node1, node2, -sign * el.mBdBasisGQFunVal[1][i] *
                                uhatEl.mBdBasisGQFunVal[2][j] ) );
                        end
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element (so second 
                        # vertex for both uhatEl and el)
                        sign = 1.0;
                        for (j, node2) in enumerate( uhatEl.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataG, ( node1, node2, -sign * el.mBdBasisGQFunVal[2][i] *
                                uhatEl.mBdBasisGQFunVal[2][j] ) );
                        end
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    else
        # no volume integrals because derivative is always zero

        # do edge integrals
        for ( i, vert ) in enumerate( aggDgMesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # do nothing because uhat = g_D
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # uhat = u_h, u on the element itself
                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        sign = -1.0;
                        push!( dataG, ( el.mNodesInd[1], el.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[1][1] * el.mBdBasisGQFunVal[1][1] ) );
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        sign = 1.0;
                        push!( dataG, ( el.mNodesInd[1], el.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[2][1] * el.mBdBasisGQFunVal[2][1] ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                else
                    error( "Boundary vertex is not included in the boundary condition." )
                end
            else
                # first, get element which corresponds to uhat for this vertex (uhat = u_L)
                S = aggDgMesh.mSwitch[i];
                uhatEl = aggDgMesh.mElements[ vert.mFaces[S] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    el = aggDgMesh.mElements[k];
                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element (so second 
                        # vertex for uhatEl, but first vertex for el)
                        sign = -1.0;
                        push!( dataG, ( el.mNodesInd[1], uhatEl.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[1][1] * uhatEl.mBdBasisGQFunVal[2][1] ) );
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element (so second 
                        # vertex for both uhatEl and el)
                        sign = 1.0;
                        push!( dataG, ( el.mNodesInd[1], uhatEl.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[2][1] * uhatEl.mBdBasisGQFunVal[2][1] ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    end

    return sp.sparse( (x->x[1]).(dataG), (x->x[2]).(dataG), (x->x[3]).(dataG), 
        aggDgMesh.mNumNodes, aggDgMesh.mNumNodes );
end

function divergence( aggDgMesh::AgglomeratedDgMesh1, baseMesh::Union{CgMesh, DgMesh}, 
    bdCond::BoundaryCondition )

    dataD = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    if aggDgMesh.mP >= 1
        # do volume integrals for divergence
        for el in aggDgMesh.mElements
            temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
            for (k, baseElInd) in enumerate( el.mBaseElementInds )
                baseEl = baseMesh.mElements[ baseElInd ];
                for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( aggDgMesh.mGaussQuadNodes )
                    temp[i,j] += baseEl.mJacobian * aggDgMesh.mGaussQuadWeights[l] * 
                        el.mBasisDerivVal[i] * el.mBasisGQFunVal[k][l,j];
                end
            end

            for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                push!(dataD, ( node1, node2, temp[i,j] ) );
            end
        end

        # do edge integrals
        for ( i, vert ) in enumerate( aggDgMesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # use qhat = q_h +/- CDir(u_h - g_D)
                    # only contribution here is qhat = q_h
                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        sign = -1.0;
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataD, ( node1, node2, -sign * el.mBdBasisGQFunVal[1][i] *
                                el.mBdBasisGQFunVal[1][j] ) );
                        end
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        sign = 1.0;
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataD, ( node1, node2, -sign * el.mBdBasisGQFunVal[2][i] *
                                el.mBdBasisGQFunVal[2][j] ) );
                        end
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # do nothing becauase qhat = g_N
                else
                    error( "Boundary vertex is not included in the boundary condition." );
                end
            else
                # first, get element which corresponds to qhat for this vertex (qhat = q_R)
                S = aggDgMesh.mSwitch[i]%2 + 1;
                qhatEl = aggDgMesh.mElements[ vert.mFaces[S] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    el = aggDgMesh.mElements[k];
                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element (so first 
                        # vertex for both quhatEl and el)
                        sign = -1.0;
                        for (j, node2) in enumerate( qhatEl.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataD, ( node1, node2, -sign * el.mBdBasisGQFunVal[1][i] *
                                qhatEl.mBdBasisGQFunVal[1][j] ) );
                        end
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element (so first 
                        # vertex for qhatEl, but second vertex for el)
                        sign = 1.0;
                        for (j, node2) in enumerate( qhatEl.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataD, ( node1, node2, -sign * el.mBdBasisGQFunVal[2][i] *
                                qhatEl.mBdBasisGQFunVal[1][j] ) );
                        end
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    else
        # no volume integrals because derivative is always zero

        # do edge integrals
        for ( i, vert ) in enumerate( aggDgMesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # use qhat = q_h +/- CDir(u_h - g_D)
                    # only contribution here is qhat = q_h
                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        sign = -1.0;
                        push!( dataD, ( el.mNodesInd[1], el.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[1][1] * el.mBdBasisGQFunVal[1][1] ) );
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        sign = 1.0;
                        push!( dataD, ( el.mNodesInd[1], el.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[2][1] * el.mBdBasisGQFunVal[2][1] ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # do nothing becauase qhat = g_N
                else
                    error( "Boundary vertex is not included in the boundary condition." );
                end
            else
                # first, get element which corresponds to qhat for this vertex (qhat = q_R)
                S = aggDgMesh.mSwitch[i]%2 + 1;
                qhatEl = aggDgMesh.mElements[ vert.mFaces[S] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    el = aggDgMesh.mElements[k];
                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element (so first 
                        # vertex for both quhatEl and el)
                        sign = -1.0;
                        push!( dataD, ( el.mNodesInd[1], qhatEl.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[1][1] * qhatEl.mBdBasisGQFunVal[1][1] ) );
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element (so first 
                        # vertex for qhatEl, but second vertex for el)
                        sign = 1.0;
                        push!( dataD, ( el.mNodesInd[1], qhatEl.mNodesInd[1], -sign * 
                            el.mBdBasisGQFunVal[2][1] * qhatEl.mBdBasisGQFunVal[1][1] ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    end

    return sp.sparse( (x->x[1]).(dataD), (x->x[2]).(dataD), (x->x[3]).(dataD), 
        aggDgMesh.mNumNodes, aggDgMesh.mNumNodes );
end

function C_matrix( aggDgMesh::AgglomeratedDgMesh1, bdCond::BoundaryCondition, 
    CDir::Float64 )

    dataC = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    if aggDgMesh.mP >= 1
        # do edge integrals
        for ( i, vert ) in enumerate( aggDgMesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # use qhat = q_h +/- CDir(u_h - g_D)
                    # only contribution here is qhat = +/- CDir * u_h

                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataC, ( node1, node2, CDir * el.mBdBasisGQFunVal[1][i] *
                                el.mBdBasisGQFunVal[1][j] ) );
                        end
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                            push!(dataC, ( node1, node2, CDir * el.mBdBasisGQFunVal[2][i] *
                                el.mBdBasisGQFunVal[2][j] ) );
                        end
                    else
                        error( "Either the vertex and element don't match or the element has 
                                    more than 2 vertices." );
                    end
                elseif vert.mIndex in bdCond.mNeuNodes
                    # do nothing
                else
                    error( "Boundary vertex is not included in the boundary condition." );
                end
            else
                # do nothing
            end
        end
    else
        # do edge integrals
        for ( i, vert ) in enumerate( aggDgMesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # use qhat = q_h +/- CDir(u_h - g_D)
                    # only contribution here is qhat = +/- CDir * u_h

                    el = aggDgMesh.mElements[ vert.mFaces[1] ];

                    if vert == el.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        # use qhat = q_h + CDir(u_h - g_D)
                        push!( dataC, ( el.mNodesInd[1], el.mNodesInd[1], CDir * 
                            el.mBdBasisGQFunVal[1][1] * el.mBdBasisGQFunVal[1][1] ) );
                    elseif vert == el.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        # use qhat = q_h - CDir(u_h - g_D)
                        push!( dataCC, ( el.mNodesInd[1], el.mNodesInd[1], CDir * 
                            el.mBdBasisGQFunVal[2][1] * el.mBdBasisGQFunVal[2][1] ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                                    more than 2 vertices." );
                    end
                elseif vert.mIndex in bdCond.mNeuNodes
                    # do nothing
                else
                    error( "Boundary vertex is not included in the boundary condition." );
                end
            else
                # do nothing
            end
        end
    end

    return sp.sparse( (x->x[1]).(dataC), (x->x[2]).(dataC), (x->x[3]).(dataC), 
        aggDgMesh.mNumNodes, aggDgMesh.mNumNodes );
end