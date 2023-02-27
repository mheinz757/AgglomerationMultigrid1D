struct CgElement <: AbstractElement
    mIndex::Int64
    mP::Int64

    mNodesInd::Vector{Int64}
    mNodesX::Vector{Float64} # Matrix in 2d
    
    mRefMap::Function # map from reference element to this element
    mJacobian::Float64 # need both inverse transpose of matrix and determinant in 2d
end

struct CgMesh <: AbstractMesh
    mP::Int64
    mElements::Vector{CgElement}
    mNumNodes::Int64
    mRefEl::ReferenceElement

    mMassMatrix::sp.SparseMatrixCSC{Float64, Int64}
    mMassMatrixLU::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}
end

############################################################################################
# CgElement creation
############################################################################################

function CgElement( face::Face, mP, vertCounter::Int64, refEl::ReferenceElement )

    mIndex = face.mIndex;

    h = face.mVertices[2].mX - face.mVertices[1].mX;
    xc = (face.mVertices[1].mX + face.mVertices[2].mX)/2.0;
    mJacobian = h/2.0;
    mRefMap(xi) = xc + h/2.0*xi;

    mNodesInd = zeros(Int64, mP + 1);
    mNodesX = zeros(Float64, mP + 1);
    for i = 1:2
        mNodesInd[i] = face.mVertices[i].mIndex;
        mNodesX[i] = face.mVertices[i].mX;
    end
    for i = 3:mP+1
        mNodesInd[i] = vertCounter;
        mNodesX[i] = mRefMap(refEl.mNodesX[i]);
        vertCounter += 1;
    end

    return CgElement( mIndex, mP, mNodesInd, mNodesX, mRefMap, mJacobian );
end

############################################################################################
# CgMesh creation
############################################################################################

function CgMesh( mesh::Mesh, mP )

    mRefEl = ReferenceElement( mP );
    mElements = Vector{CgElement}( undef, length(mesh.mFaces) );

    vertCounter = length(mesh.mVertices) + 1;
    for (i, face) in enumerate(mesh.mFaces)
        mElements[i] = CgElement( face, mP, vertCounter, mRefEl );
        vertCounter += mP - 1;
    end

    mNumNodes = vertCounter - 1;

    data = Vector{Tuple{Int64, Int64, Float64}}(undef, 0);

    for el in mElements
        for (j, node2) in enumerate(el.mNodesInd), (i, node1) in enumerate(el.mNodesInd)
            push!( data, ( node1, node2, el.mJacobian * mRefEl.mMassMatrix[i,j] ) );
        end
    end

    mMassMatrix = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        mNumNodes, mNumNodes );
    mMassMatrixLU = la.lu(mMassMatrix);

    return CgMesh( mP, mElements, mNumNodes, mRefEl, mMassMatrix, mMassMatrixLU );
end

############################################################################################
# operators and rhs for CgMesh
############################################################################################

# function to initialize stiffness matrix for a CgMesh
function cg_stiffness( cgMesh::CgMesh, bdCond::BoundaryCondition )
    # need to be able to integrate derivatives of the basis function over each element
    # # this requires jacobian, gauss quad nodes and weights, and the values of the 
    # # derivatives of basis functions at each of those things
    # also need boundary information for boundary conditions
    # # need to know which vertices are Dirichlet vertices
    
    n = length( cgMesh.mElements );
    refEl = cgMesh.mRefEl;
    dirNodes = bdCond.mDirNodes;

    # do volume integrals for stiffness
    data = Vector{Tuple{Int64, Int64, Float64}}( undef, 0 );
    
    for el in cgMesh.mElements
        temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
        for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
            temp[i,j] += (1.0 / el.mJacobian) * refEl.mGaussQuadWeights[l] * 
                refEl.mBasisGQDerivVal[l,i] * refEl.mBasisGQDerivVal[l,j];
        end

        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
            push!(data, ( node1, node2, temp[i,j] ) )
        end
    end

    A = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        cgMesh.mNumNodes, cgMesh.mNumNodes );

    # adjust matrix for strong dirichlet boundaries
    A[ dirNodes, : ] = sp.spzeros( length(dirNodes), size(A,2) );
    A[ :, dirNodes ] = sp.spzeros( size(A,1), length(dirNodes) );
    A[ dirNodes, dirNodes ] = sp.sparse( la.I, length(dirNodes), length(dirNodes) );

    return A;
end

# function to initialize stiffness matrix and rhs for CgMesh
function cg_stiffness_and_rhs( cgMesh::CgMesh, mesh::Mesh, func::Function,  
    bdCond::BoundaryCondition )
    # need to be able to integrate derivatives of the basis function over each element and 
    # func against basis function over each element
    # # this requires jacobian, gauss quad nodes and weights, and the values of the 
    # # basis functions and their derivatives at each of those things
    # also need boundary information for boundary conditions
    # # need to know which vertices are Dirichlet vertices, and what the Dirichlet values 
    # # are there
    # # need to know which edges are Neumann edges, and have the bdCond structure to 
    # # say what the Neumann function is on that edge (vertices in 1d still)
    
    n = length( cgMesh.mElements );
    refEl = cgMesh.mRefEl;
    data = Vector{Tuple{Int64, Int64, Float64}}( undef, 0 );
    f = zeros( cgMesh.mNumNodes )

    # do volume integrals for stiffness matrix and rhs
    for el in cgMesh.mElements
        temp = zeros( length( el.mNodesInd ), length( el.mNodesInd ) );
        for j = 1:length( el.mNodesInd ), i = 1:length( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
            temp[i,j] += (1.0 / el.mJacobian) * refEl.mGaussQuadWeights[l] * 
                refEl.mBasisGQDerivVal[l,i] * refEl.mBasisGQDerivVal[l,j];
        end

        for (j, node2) in enumerate( el.mNodesInd ), (i, node1) in enumerate( el.mNodesInd ) 
            push!(data, ( node1, node2, temp[i,j] ) )
        end

        for (i, node) in enumerate( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
            f[node] += el.mJacobian * refEl.mGaussQuadWeights[l] * 
                refEl.mBasisGQFunVal[l,i] * func( el.mRefMap(refEl.mGaussQuadNodes[l]) );
        end
    end

    A = sp.sparse( (x->x[1]).(data), (x->x[2]).(data), (x->x[3]).(data), 
        cgMesh.mNumNodes, cgMesh.mNumNodes );

    # adjust rhs because of neumann boundaries
    for k in bdCond.mNeuNodes
        vert = mesh.mVertices[k];
        face = mesh.mFaces[vert.mFaces[1]];
        if vert == face.mVertices[1]
            # left vertex of element
            f[vert.mIndex] += -bdCond.mBdCond[-vert.mFaces[2]][2];
        else
            # right vertex of element
            f[vert.mIndex] += bdCond.mBdCond[-vert.mFaces[2]][2];
        end
    end 

    # adjust matrix and rhs for strong dirichlet boundaries
    f += -A[:, bdCond.mDirNodes] * bdCond.mDirVals; 
    f[ bdCond.mDirNodes ] = bdCond.mDirVals;
    A[ bdCond.mDirNodes, : ] = sp.spzeros( length(bdCond.mDirNodes), size(A,2) );
    A[ :, bdCond.mDirNodes ] = sp.spzeros( size(A,1), length(bdCond.mDirNodes) );
    A[ bdCond.mDirNodes, bdCond.mDirNodes ] = sp.sparse( la.I, length(bdCond.mDirNodes), 
        length(bdCond.mDirNodes) );

    return A, f;
end

# function to compute rhs corresponding to cg_mesh
function cg_rhs( cgMesh::CgMesh, mesh::Mesh, func::Function, 
    bdCond::BoundaryCondition )
    # need to be able to integrate func against basis function over each element
    # # this requires jacobian, gauss quad nodes and weights, and the values of the 
    # # basis functions at each of those things
    # also need boundary information for boundary conditions
    # # need to know which vertices are Dirichlet vertices, and what the Dirichlet values 
    # # are there
    # # need to know which edges are Neumann edges, and have the bdCond structure to 
    # # say what the Neumann function is on that edge (vertices in 1d still)
    
    n = length( cgMesh.mElements );
    refEl = cgMesh.mRefEl
    f = zeros( cgMesh.mNumNodes );

    for (k, el) in enumerate( cgMesh.mElements )
        for (i, vert) in enumerate( el.mNodesInd ), l = 1:length( refEl.mGaussQuadNodes )
            f[vert] += el.mJacobian * refEl.mGaussQuadWeights[l] * 
                refEl.mBasisGQFunVal[l,i] * func( el.mRefMap(refEl.mGaussQuadNodes[l]) ); 
        end

        face = mesh.mFaces[k];
        # go through boundary faces to adjust rhs for strong dirichlet boundaries
        if isBoundary( face )
            # go through nodes
            for (i, vert1) in enumerate( face.mVertices )
                # check if the node is a Dirichlet node
                if isBoundary( vert1 ) && bdCond.mBdCond[-vert1.mFaces[2]][1] == :dir
                    # update all other nodes
                    for (j, node2) in enumerate( el.mNodesInd )
                        temp = 0.0;
                        for l = 1:length( refEl.mGaussQuadNodes )
                            temp += (1.0 / el.mJacobian) * refEl.mGaussQuadWeights[l] * 
                                refEl.mBasisGQDerivVal[l,i] * refEl.mBasisGQDerivVal[l,j];
                        end
                        f[node2] += -temp * bdCond.mBdCond[-vert1.mFaces[2]][2]
                    end
                end
            end
        end
    end

    # adjust rhs because of neumann boundaries
    for k in bdCond.mNeuNodes
        vert = mesh.mVertices[k];
        face = mesh.mFaces[vert.mFaces[1]];
        if vert == face.mVertices[1]
            # left vertex of element
            f[vert.mIndex] += -bdCond.mBdCond[-vert.mFaces[2]][2];
        else
            # right vertex of element
            f[vert.mIndex] += bdCond.mBdCond[-vert.mFaces[2]][2];
        end
    end 

    # set strong Dirichlet boundaries
    f[ bdCond.mDirNodes ] = bdCond.mDirVals;

    return f;
end

