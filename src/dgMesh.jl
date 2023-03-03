struct DgElement <: AbstractElement
    mIndex::Int64
    mP::Int64

    mNodesInd::Vector{Int64}
    mNodesX::Vector{Float64} # matrix in 2d
    
    mRefMap::Function # map from reference element to this element
    mJacobian::Float64 # need both inverse transpose of matrix and determinant in 2d

    mNeighborVertInd::Vector{Int64} # for each edge associated with this element, stores the 
                                    # local index of the edge in the neighboring element 
                                    # that shares the edge
end

struct DgMesh <: AbstractMesh
    mP::Int64
    mElements::Vector{DgElement}
    mNumNodes::Int64
    mRefEl::ReferenceElement
    # also would have a reference edge in 2d
    mSwitch::Vector{Int64} # switch function for each vertex (edge in 2d)

    mMassMatrix::BlockDiagonal # can switch to bd.BlockDiagonal{Float64, Matrix{Float64}}
    mMassMatrixLU::BlockDiagonalLU
end

############################################################################################
# DgElement creation
############################################################################################

function DgElement( face::Face, mP, refEl::ReferenceElement )

    mIndex = face.mIndex;

    h = face.mVertices[2].mX - face.mVertices[1].mX;
    xc = (face.mVertices[1].mX + face.mVertices[2].mX)/2.0;
    mJacobian = h/2.0;
    mRefMap(xi) = xc .+ h/2.0*xi;

    mNodesInd = zeros(Int64, mP + 1);
    mNodesX = zeros(Float64, mP + 1);
    for i = 1:mP+1
        mNodesInd[i] = (mIndex - 1) * (mP + 1) + i;
        mNodesX[i] = mRefMap(refEl.mNodesX[i]);
    end

    # mNeighborVertInd still unused
    mNeighborVertInd = Vector{Int64}( undef, 0 );

    return DgElement( mIndex, mP, mNodesInd, mNodesX, mRefMap, mJacobian, mNeighborVertInd );
end

############################################################################################
# DgMesh creation
############################################################################################

function DgMesh( mesh::Mesh, mP )

    mRefEl = ReferenceElement( mP );
    mElements = Vector{DgElement}( undef, length(mesh.mFaces) );

    for (i, face) in enumerate(mesh.mFaces)
        mElements[i] = DgElement( face, mP, mRefEl );
    end

    mNumNodes = length( mElements ) * (mP + 1);

    massBlocks = Vector{Matrix{Float64}}( undef, length( mElements ) );
    massBlockSize = mP + 1;
    massBlockInds = zeros( massBlockSize, length( mElements ) );

    for el in mElements
        massBlocks[el.mIndex] = el.mJacobian * mRefEl.mMassMatrix;
        massBlockInds[:, el.mIndex] = el.mNodesInd;
    end

    mMassMatrix = BlockDiagonal( massBlocks, massBlockSize, massBlockInds );
    mMassMatrixLU = lu( mMassMatrix );

    mSwitch = Vector{Int64}( undef, length( mesh.mVertices ) );

    for ( i, vert ) in enumerate( mesh.mVertices )
        if isBoundary( vert )
            # do something if the vertex is a boundary vertex in case of periodic b.c.
            face = mesh.mFaces[ vert.mFaces[1] ];

            # if the boundary vertex is on the right, select its own face
            # otherwise, select *other* face
            if vert.mX > min(face.mVertices[1].mX, face.mVertices[2].mX)
                mSwitch[i] = 1;
            else
                mSwitch[i] = 2;
            end
        else
            face1 = mesh.mFaces[ vert.mFaces[1] ];
            face2 = mesh.mFaces[ vert.mFaces[1] ];

            x1 = max(face1.mVertices[1].mX, face1.mVertices[2].mX);
            x2 = max(face2.mVertices[1].mX, face2.mVertices[2].mX);

            # select index of face that is on the left of the vertex
            if x1 > x2
                mSwitch[i] = 2; 
            else
                mSwitch[i] = 1;
            end
        end
    end

    return DgMesh( mP, mElements, mNumNodes, mRefEl, mSwitch, mMassMatrix, mMassMatrixLU );
end

function DgMesh( mesh::Mesh, mP, mSwitch )

    mRefEl = ReferenceElement( mP );
    mElements = Vector{DgElement}( undef, length(mesh.mFaces) );

    for (i, face) in enumerate(mesh.mFaces)
        mElements[i] = DgElement( face, mP, mRefEl );
    end

    mNumNodes = length( mElements ) * (mP + 1);

    massBlocks = Vector{Matrix{Float64}}( undef, length( mElements ) );
    massBlockSize = mP + 1;
    massBlockInds = zeros( massBlockSize, length( mElements ) );

    for el in mElements
        massBlocks[el.mIndex] = el.mJacobian * mRefEl.mMassMatrix;
        massBlockInds[:, el.mIndex] = el.mNodesInd;
    end

    mMassMatrix = BlockDiagonal( massBlocks, massBlockSize, massBlockInds );
    mMassMatrixLU = lu( mMassMatrix );

    return DgMesh( mP, mElements, mNumNodes, mRefEl, mSwitch, mMassMatrix, mMassMatrixLU );
end

############################################################################################
# operators for DgMesh
############################################################################################

function dg_flux_operators( dgMesh::DgMesh, mesh::Mesh, bdCond::BoundaryCondition, 
    CDir::Float64 )

    refEl = dgMesh.mRefEl;

    dataG = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    dataD = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);
    dataC = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    if dgMesh.mP >= 1
        # do volume integrals for gradient
        for el in dgMesh.mElements
            temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
            for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
                temp[i,j] += refEl.mGaussQuadWeights[l] * refEl.mBasisGQDerivVal[l,i] * 
                    refEl.mBasisGQFunVal[l,j];
            end

            for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                push!( dataG, ( node1, node2, temp[i,j] ) );
                push!( dataD, ( node1, node2, temp[i,j] ) );
            end
        end

        # do edge integrals
        for ( i, vert ) in enumerate( mesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # Gradient: do nothing because uhat = g_D
                    # Divergence: use qhat = q_h +/- CDir(u_h - g_D)
                    ### only contribution here is qhat = q_h
                    # C matrix: use qhat = q_h +/- CDir(u_h - g_D)
                    ### only contribution here is qhat = +/- CDir u_h

                    meshEl = mesh.mFaces[ vert.mFaces[1] ];
                    dgEl = dgMesh.mElements[ vert.mFaces[1] ];

                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        # phi_i at xin is only nonzero when i is the first vertex in dgEl
                        push!( dataD, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], 1.0 ) );
                        push!( dataC, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], CDir ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        # phi_i at xout is only nonzero when i is the second vertex in dgEl
                        push!( dataD, ( dgEl.mNodesInd[2], dgEl.mNodesInd[2], -1.0 ) );
                        push!( dataC, ( dgEl.mNodesInd[2], dgEl.mNodesInd[2], CDir ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end

                elseif vert.mIndex in bdCond.mNeuNodes 
                    # Gradient: uhat = u_h, u on the element itself
                    # Divergence: do nothing because qhat = g_N
                    # C_matrix: do nothing

                    meshEl = mesh.mFaces[ vert.mFaces[1] ];
                    dgEl = dgMesh.mElements[ vert.mFaces[1] ];

                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        # phi_i at xin is only nonzero when i is the first vertex in dgEl
                        push!( dataG, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        # phi_i at xout is only nonzero when i is the second vertex in dgEl
                        push!( dataG, ( dgEl.mNodesInd[2], dgEl.mNodesInd[2], -1.0 ) );
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
                S = dgMesh.mSwitch[i];
                uhatEl = dgMesh.mElements[ vert.mFaces[S] ];
                qhatEl = dgMesh.mElements[ vert.mFaces[S%2 + 1] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    meshEl = mesh.mFaces[k];
                    dgEl = dgMesh.mElements[k];
                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        # uhat at xin is just the value of u at the second vertex in uhatEl
                        # qhat at xin is just the value of q at the first vertex in qhatEl
                        # phi_i at xin is only nonzero when i is the first vertex in dgEl
                        push!( dataG, ( dgEl.mNodesInd[1], uhatEl.mNodesInd[2], 1.0 ) );
                        push!( dataD, ( dgEl.mNodesInd[1], qhatEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        # uhat at xout is just the value of u at the second vertex in uhatEl
                        # qhat at xout is just the value of u at the first vertex in uhatEl
                        # phi_i at xout is only nonzero when i is the second vertex in dgEl
                        push!( dataG, ( dgEl.mNodesInd[2], uhatEl.mNodesInd[2], -1.0 ) );
                        push!( dataD, ( dgEl.mNodesInd[2], qhatEl.mNodesInd[1], -1.0 ) );
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
        for ( i, vert ) in enumerate( mesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # Gradient: do nothing because uhat = g_D
                    # Divergence: use qhat = q_h +/- CDir(u_h - g_D)
                    ### only contribution here is qhat = q_h
                    # C matrix: use qhat = q_h +/- CDir(u_h - g_D)
                    ### only contribution here is qhat = +/- CDir u_h

                    meshEl = mesh.mFaces[ vert.mFaces[1] ];
                    dgEl = dgMesh.mElements[ vert.mFaces[1] ];

                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        push!( dataD, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], 1.0 ) );
                        push!( dataC, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], CDir ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        push!( dataD, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], -1.0 ) );
                        push!( dataC, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], CDir ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # Gradient: uhat = u_h, u on the element itself
                    # Divergence: do nothing because qhat = g_N
                    # C_matrix: do nothing

                    meshEl = mesh.mFaces[ vert.mFaces[1] ];
                    dgEl = dgMesh.mElements[ vert.mFaces[1] ];

                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        push!( dataG, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        push!( dataG, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], -1.0 ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                else
                    error( "Boundary vertex is not included in the boundary condition." )
                end
            else
                # first, get elements which correspond to uhat and qhat for this vertex 
                # uhat = u_L, qhat = q_R
                S = dgMesh.mSwitch[i];
                uhatEl = dgMesh.mElements[ vert.mFaces[S] ];
                qhatEl = dgMesh.mElements[ vert.mFaces[S%2 + 1] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    meshEl = mesh.mFaces[k];
                    dgEl = dgMesh.mElements[k];
                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        push!( dataG, ( dgEl.mNodesInd[1], uhatEl.mNodesInd[1], 1.0 ) );
                        push!( dataD, ( dgEl.mNodesInd[1], qhatEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        push!( dataG, ( dgEl.mNodesInd[1], uhatEl.mNodesInd[1], -1.0 ) );
                        push!( dataD, ( dgEl.mNodesInd[1], qhatEl.mNodesInd[1], -1.0 ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    end

    G = sp.sparse( (x->x[1]).(dataG), (x->x[2]).(dataG), (x->x[3]).(dataG), 
        dgMesh.mNumNodes, dgMesh.mNumNodes );
    D = sp.sparse( (x->x[1]).(dataD), (x->x[2]).(dataD), (x->x[3]).(dataD), 
        dgMesh.mNumNodes, dgMesh.mNumNodes );
    C = sp.sparse( (x->x[1]).(dataC), (x->x[2]).(dataC), (x->x[3]).(dataC), 
        dgMesh.mNumNodes, dgMesh.mNumNodes );

    return G, D, C;
end

############################################################################################
# rhs for DgMesh
############################################################################################

function dg_flux_rhs( dgMesh::DgMesh, mesh::Mesh, func::Function, bdCond::BoundaryCondition, 
    CDir::Float64 )

    f = zeros( dgMesh.mNumNodes );
    r = zeros( dgMesh.mNumNodes );

    refEl = dgMesh.mRefEl;

    # do volume integrals
    for el in dgMesh.mElements
        for (i, node) in enumerate( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
            f[node] += el.mJacobian * refEl.mGaussQuadWeights[l] * 
                refEl.mBasisGQFunVal[l,i] * func( el.mRefMap( refEl.mGaussQuadNodes[l] ) ); 
        end
    end

    if dgMesh.mP >= 1
        # do edge integrals for dirichlet boundaries
        for ( i, nodeIdx ) in enumerate( bdCond.mDirNodes )
            vert = mesh.mVertices[ nodeIdx ];
            dirVal = bdCond.mDirVals[i];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # use qhat = q_h + CDir(u_h - g_D)
                # phi_i at xin is only nonzero when i is the first vertex in dgEl
                f[ dgEl.mNodesInd[1] ] += CDir * dirVal;
                r[ dgEl.mNodesInd[1] ] += -dirVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # use qhat = q_h - CDir(u_h - g_D)
                # phi_i at xout is only nonzero when i is the second vertex in dgEl
                f[ dgEl.mNodesInd[2] ] += CDir * dirVal;
                r[ dgEl.mNodesInd[2] ] += dirVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end

        # do edge integrals for neumann boundaries
        for nodeIdx in bdCond.mNeuNodes
            vert = mesh.mVertices[ nodeIdx ];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # phi_i at xin is only nonzero when i is the first vertex in dgEl

                neuVal = bdCond.mBdCond[1][2];
                f[ dgEl.mNodesInd[1] ] += -neuVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # phi_i at xout is only nonzero when i is the second vertex in dgEl

                neuVal = bdCond.mBdCond[2][2];
                f[ dgEl.mNodesInd[2] ] += neuVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end
    else
        # do edge integrals for dirichlet boundaries
        for ( i, nodeIdx ) in enumerate( bdCond.mDirNodes )
            vert = mesh.mVertices[ nodeIdx ];
            dirVal = bdCond.mDirVals[i];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # use qhat = q_h + CDir(u_h - g_D)
                f[ dgEl.mNodesInd[1] ] += CDir * dirVal;
                r[ dgEl.mNodesInd[1] ] += -dirVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # use qhat = q_h - CDir(u_h - g_D)
                f[ dgEl.mNodesInd[1] ] += CDir * dirVal;
                r[ dgEl.mNodesInd[1] ] += dirVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end

        # do edge integrals for neumann boundaries
        for nodeIdx in bdCond.mNeuNodes
            vert = mesh.mVertices[ nodeIdx ];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                neuVal = bdCond.mBdCond[1][2];
                f[ dgEl.mNodesInd[1] ] += -neuVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                neuVal = bdCond.mBdCond[2][2];
                f[ dgEl.mNodesInd[1] ] += neuVal;
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
# operators for DgMesh
############################################################################################

function gradient( dgMesh::DgMesh, mesh::Mesh, bdCond::BoundaryCondition )

    refEl = dgMesh.mRefEl;

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    if dgMesh.mP >= 1
        # do volume integrals for gradient
        for el in dgMesh.mElements
            temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
            for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
                temp[i,j] += refEl.mGaussQuadWeights[l] * refEl.mBasisGQDerivVal[l,i] * 
                    refEl.mBasisGQFunVal[l,j];
            end

            for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                push!(data, ( node1, node2, temp[i,j] ) );
            end
        end

        # do edge integrals
        for ( i, vert ) in enumerate( mesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # do nothing because uhat = g_D
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # uhat = u_h, u on the element itself
                    meshEl = mesh.mFaces[ vert.mFaces[1] ];
                    dgEl = dgMesh.mElements[ vert.mFaces[1] ];

                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        # phi_i at xin is only nonzero when i is the first vertex in dgEl
                        push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        # phi_i at xout is only nonzero when i is the second vertex in dgEl
                        push!( data, ( dgEl.mNodesInd[2], dgEl.mNodesInd[2], -1.0 ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                else
                    error( "Boundary vertex is not included in the boundary condition." );
                end
            else
                # first, get element which corresponds to uhat for this vertex (uhat = u_L)
                S = dgMesh.mSwitch[i];
                uhatEl = dgMesh.mElements[ vert.mFaces[S] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    meshEl = mesh.mFaces[k];
                    dgEl = dgMesh.mElements[k];
                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        # uhat at xin is just the value of u at the second vertex in uhatEl
                        # phi_i at xin is only nonzero when i is the first vertex in dgEl
                        push!( data, ( dgEl.mNodesInd[1], uhatEl.mNodesInd[2], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        # uhat at xout is just the value of u at the second vertex in uhatEl
                        # phi_i at xout is only nonzero when i is the second vertex in dgEl
                        push!( data, ( dgEl.mNodesInd[2], uhatEl.mNodesInd[2], -1.0 ) );
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
        for ( i, vert ) in enumerate( mesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # do nothing because uhat = g_D
                elseif vert.mIndex in bdCond.mNeuNodes 
                    # uhat = u_h, u on the element itself
                    meshEl = mesh.mFaces[ vert.mFaces[1] ];
                    dgEl = dgMesh.mElements[ vert.mFaces[1] ];

                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], -1.0 ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                else
                    error( "Boundary vertex is not included in the boundary condition." )
                end
            else
                # first, get element which corresponds to uhat for this vertex (uhat = u_L)
                S = dgMesh.mSwitch[i];
                uhatEl = dgMesh.mElements[ vert.mFaces[S] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    meshEl = mesh.mFaces[k];
                    dgEl = dgMesh.mElements[k];
                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        push!( data, ( dgEl.mNodesInd[1], uhatEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        push!( data, ( dgEl.mNodesInd[1], uhatEl.mNodesInd[1], -1.0 ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    end

    return sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        dgMesh.mNumNodes, dgMesh.mNumNodes );
end

function divergence( dgMesh::DgMesh, mesh::Mesh, bdCond::BoundaryCondition )

    refEl = dgMesh.mRefEl;

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    if dgMesh.mP >= 1
        # do volume integrals for divergence
        for el in dgMesh.mElements
            temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
            for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
                temp[i,j] += refEl.mGaussQuadWeights[l] * refEl.mBasisGQDerivVal[l,i] * 
                    refEl.mBasisGQFunVal[l,j];
            end

            for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
                push!(data, ( node1, node2, temp[i,j] ) );
            end
        end

        # do edge integrals
        for ( i, vert ) in enumerate( mesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # use qhat = q_h +/- CDir(u_h - g_D)
                    # only contribution here is qhat = q_h
                    meshEl = mesh.mFaces[ vert.mFaces[1] ];
                    dgEl = dgMesh.mElements[ vert.mFaces[1] ];

                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        # phi_i at xin is only nonzero when i is the first vertex in dgEl
                        push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        # phi_i at xout is only nonzero when i is the second vertex in dgEl
                        push!( data, ( dgEl.mNodesInd[2], dgEl.mNodesInd[2], -1.0 ) );
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
                S = dgMesh.mSwitch[i]%2 + 1;
                qhatEl = dgMesh.mElements[ vert.mFaces[S] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    meshEl = mesh.mFaces[k];
                    dgEl = dgMesh.mElements[k];
                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        # qhat at xin is just the value of u at the first vertex in qhatEl
                        # phi_i at xin is only nonzero when i is the first vertex in dgEl
                        push!( data, ( dgEl.mNodesInd[1], qhatEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        # qhat at xout is just the value of u at the first vertex in uhatEl
                        # phi_i at xout is only nonzero when i is the second vertex in dgEl
                        push!( data, ( dgEl.mNodesInd[2], qhatEl.mNodesInd[1], -1.0 ) );
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
        for ( i, vert ) in enumerate( mesh.mVertices )
            if isBoundary(vert)
                if vert.mIndex in bdCond.mDirNodes
                    # use qhat = q_h +/- CDir(u_h - g_D)
                    # only contribution here is qhat = q_h
                    meshEl = mesh.mFaces[ vert.mFaces[1] ];
                    dgEl = dgMesh.mElements[ vert.mFaces[1] ];

                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], -1.0 ) );
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
                S = dgMesh.mSwitch[i]%2 + 1;
                qhatEl = dgMesh.mElements[ vert.mFaces[S] ];

                # now loop over the elements adjacent to the vertex
                for k in vert.mFaces
                    meshEl = mesh.mFaces[k];
                    dgEl = dgMesh.mElements[k];
                    if vert == meshEl.mVertices[1]
                        # the vertex is the incoming boundary of this element
                        push!( data, ( dgEl.mNodesInd[1], qhatEl.mNodesInd[1], 1.0 ) );
                    elseif vert == meshEl.mVertices[2]
                        # the vertex is the outgoing boundary of this element
                        push!( data, ( dgEl.mNodesInd[1], qhatEl.mNodesInd[1], -1.0 ) );
                    else
                        error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
                    end
                end
            end
        end
    end

    return sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        dgMesh.mNumNodes, dgMesh.mNumNodes );
end

function C_matrix( dgMesh::DgMesh, mesh::Mesh, bdCond::BoundaryCondition, CDir::Float64 )

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    if dgMesh.mP >= 1
        # do edge integrals
        for i in bdCond.mDirNodes
            vert = mesh.mVertices[i];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # use qhat = q_h + CDir(u_h - g_D)
                # phi_i at xin is only nonzero when i is the first vertex in dgEl
                push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], CDir ) );
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # use qhat = q_h - CDir(u_h - g_D)
                # phi_i at xout is only nonzero when i is the second vertex in dgEl
                push!( data, ( dgEl.mNodesInd[2], dgEl.mNodesInd[2], CDir ) );
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end
    else
        # do edge integrals
        for i in bdCond.mDirNodes
            vert = mesh.mVertices[i];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # use qhat = q_h + CDir(u_h - g_D)
                push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], CDir ) );
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # use qhat = q_h - CDir(u_h - g_D)
                push!( data, ( dgEl.mNodesInd[1], dgEl.mNodesInd[1], CDir ) );
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end
    end

    return sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        dgMesh.mNumNodes, dgMesh.mNumNodes );
end

############################################################################################
# rhs for DgMesh
############################################################################################

function r_vector( dgMesh::DgMesh, mesh::Mesh, bdCond::BoundaryCondition )

    r = zeros( dgMesh.mNumNodes )

    if dgMesh.mP >= 1
        # do edge integrals
        for ( i, nodeIdx ) in enumerate( bdCond.mDirNodes )
            vert = mesh.mVertices[ nodeIdx ];
            dirVal = bdCond.mDirVals[i];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # phi_i at xin is only nonzero when i is the first vertex in dgEl
                r[ dgEl.mNodesInd[1] ] += -dirVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # phi_i at xout is only nonzero when i is the second vertex in dgEl
                r[ dgEl.mNodesInd[2] ] += dirVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end
    else
        # do edge integrals
        for ( i, nodeIdx ) in enumerate( bdCond.mDirNodes )
            vert = mesh.mVertices[ nodeIdx ];
            dirVal = bdCond.mDirVals[i];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                r[ dgEl.mNodesInd[1] ] += -dirVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                r[ dgEl.mNodesInd[1] ] += dirVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end
    end

    return r;
end

function f_vector( dgMesh::DgMesh, mesh::Mesh, func::Function, bdCond::BoundaryCondition, 
    CDir::Float64 )

    f = zeros( dgMesh.mNumNodes );
    refEl = dgMesh.mRefEl;

    # do volume integrals
    for el in dgMesh.mElements
        for (i, node) in enumerate( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
            f[node] += el.mJacobian * refEl.mGaussQuadWeights[l] * 
                refEl.mBasisGQFunVal[l,i] * func( el.mRefMap( refEl.mGaussQuadNodes[l] ) ); 
        end
    end

    if dgMesh.mP >= 1
        # do edge integrals for dirichlet boundaries
        for ( i, nodeIdx ) in enumerate( bdCond.mDirNodes )
            vert = mesh.mVertices[ nodeIdx ];
            dirVal = bdCond.mDirVals[i];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # use qhat = q_h + CDir(u_h - g_D)
                # phi_i at xin is only nonzero when i is the first vertex in dgEl
                f[ dgEl.mNodesInd[1] ] += CDir * dirVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # use qhat = q_h - CDir(u_h - g_D)
                # phi_i at xout is only nonzero when i is the second vertex in dgEl
                f[ dgEl.mNodesInd[2] ] += CDir * dirVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end

        # do edge integrals for neumann boundaries
        for nodeIdx in bdCond.mNeuNodes
            vert = mesh.mVertices[ nodeIdx ];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # phi_i at xin is only nonzero when i is the first vertex in dgEl

                neuVal = bdCond.mBdCond[1][2];
                f[ dgEl.mNodesInd[1] ] += -neuVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # phi_i at xout is only nonzero when i is the second vertex in dgEl

                neuVal = bdCond.mBdCond[2][2];
                f[ dgEl.mNodesInd[2] ] += neuVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end
    else
        # do edge integrals for dirichlet boundaries
        for ( i, nodeIdx ) in enumerate( bdCond.mDirNodes )
            vert = mesh.mVertices[ nodeIdx ];
            dirVal = bdCond.mDirVals[i];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                # use qhat = q_h + CDir(u_h - g_D)
                f[ dgEl.mNodesInd[1] ] += CDir * dirVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                # use qhat = q_h - CDir(u_h - g_D)
                f[ dgEl.mNodesInd[1] ] += CDir * dirVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end

        # do edge integrals for neumann boundaries
        for nodeIdx in bdCond.mNeuNodes
            vert = mesh.mVertices[ nodeIdx ];

            meshEl = mesh.mFaces[ vert.mFaces[1] ];
            dgEl = dgMesh.mElements[ vert.mFaces[1] ];

            if vert == meshEl.mVertices[1]
                # the vertex is the incoming boundary of this element
                neuVal = bdCond.mBdCond[1][2];
                f[ dgEl.mNodesInd[1] ] += -neuVal;
            elseif vert == meshEl.mVertices[2]
                # the vertex is the outgoing boundary of this element
                neuVal = bdCond.mBdCond[2][2];
                f[ dgEl.mNodesInd[1] ] += neuVal;
            else
                error( "Either the vertex and element don't match or the element has 
                            more than 2 vertices." );
            end
        end
    end

    return f;
end