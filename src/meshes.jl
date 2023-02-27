"""
    Vertex

Stores information about a vertex on a 1-dimensional grid

# Fields
- `mIndex::Int64`
- `mX::Float64`: X position
- `mFaces::Vector{Int64}`: vector of indices of faces that vertex belongs to
"""
struct Vertex
    mIndex::Int64
    mX::Float64
    mFaces::Vector{Int64}

    Vertex( mIndex, mX, mFaces = zeros(Int64, 2) ) = new( mIndex, mX, mFaces );
end

"""
    Face

Stores information about a face on a 1-dimensional grid

# Fields
- `mIndex::Int64`
- `mVertices::Vector{Vertex}`: vector of pointers to vertices of the face
- `mNeighbors::Vector{Int64}`: vector of indices of neighbor faces
"""
struct Face
    mIndex::Int64
    mVertices::Vector{Vertex}
    mNeighbors::Vector{Int64}

    Face( mIndex, mVertices, mNeighbors ) = new( mIndex, mVertices, mNeighbors );
    Face( mIndex, nV::Int64 ) = new( mIndex, Vector{Vertex}( undef, nV ), 
        zeros(Int64, nV) );
end

"""
    Mesh

Stores information about a 1-dimensional mesh

# Fields
- `mVertices::Vector{Vertex}`: list of pointers to vertices of the mesh
- `mFaces::Vector{Face}`: list of pointers to faces of the mesh
"""
struct Mesh
    mVertices::Vector{Vertex}
	mFaces::Vector{Face}
end

"""
    isBoundary( vertex::Vertex )

Determines if a vertex is a boundary vertex of the 1-dimensional mesh
"""
function isBoundary( vertex::Vertex )
    return vertex.mFaces[2] < 1;
end

"""
    isBoundary( face::Face )

Determines if a face is a boundary element of the 1-dimensional mesh
"""
function isBoundary( face::Face )
	return face.mNeighbors[end] == 0;
end