include("../src/AgglomerationMultigrid1D.jl")

import .AgglomerationMultigrid1D as aggmg

function create_uniform_mesh( n, xin, xout )
    h = (xout-xin)/n;

    faces = Vector{aggmg.Face}(undef, n);
    vertices = Vector{aggmg.Vertex}(undef, n+1);

    vertices[1] = aggmg.Vertex(1, xin);
    for i = 1:n
        vertices[i+1] = aggmg.Vertex( i+1, xin + (i/n) * (xout - xin) );
        faces[i] = aggmg.Face(i, 2);
        for j = 1:2
            faces[i].mVertices[j] = vertices[i-1+j]
        end
    end

    for i = 1:n
        cFace = faces[i];
        for j = 1:2
            cVertex = cFace.mVertices[j];
            if cVertex.mFaces[1] == 0
                cVertex.mFaces[1] = cFace.mIndex;
            elseif cVertex.mFaces[2] == 0
                cVertex.mFaces[2] = cFace.mIndex;
            else
                error( "Vertex can only neighbor two faces." );
            end
        end
    end

    #Deal with face neighbor lists
    adjFaces = Dict{Int64,Int64}();
    for cFace in faces
        empty!( adjFaces );
        #Find all possible adjacencies
        for fVert in cFace.mVertices
            for face in fVert.mFaces
                if haskey( adjFaces, face )
                    adjFaces[face] += 1;
                else
                    push!( adjFaces, face => 1 );
                end
            end
        end

        #Figure out which faces are really adjacent
        nIndex = 1;
        for ( f, count ) in adjFaces
            if f != 0 && f != cFace.mIndex && count == 1
                cFace.mNeighbors[nIndex] = f; nIndex += 1;
            end
        end
    end

    return aggmg.Mesh( vertices, faces );
end

function set_boundary!( mesh::aggmg.Mesh, xin, xout, bdCond::Vector{ Tuple{Symbol, Float64} } )

    dirNodes = Vector{Int64}( undef, 0 );
    dirVals = Vector{Float64}( undef, 0 );
    neuNodes = Vector{Int64}( undef, 0 );

    for (k, face) in enumerate(mesh.mFaces)

        if aggmg.isBoundary(face)
            for vert in face.mVertices
                if aggmg.isBoundary(vert) && ( abs( vert.mX - xin ) < 1e-15 )
                    vert.mFaces[2] = -1;
                    if bdCond[1][1] == :dir
                        push!( dirNodes, vert.mIndex )
                        push!( dirVals, bdCond[1][2] )
                    elseif bdCond[1][1] == :neu
                        push!( neuNodes, vert.mIndex )
                    end
                elseif aggmg.isBoundary(vert) && ( abs( vert.mX - xout ) < 1e-15 )
                    vert.mFaces[2] = -2;
                    if bdCond[2][1] == :dir
                        push!( dirNodes, vert.mIndex )
                        push!( dirVals, bdCond[2][2] )
                    elseif bdCond[2][1] == :neu
                        push!( neuNodes, vert.mIndex )
                    end
                end
            end
        end
    end
            
    return aggmg.BoundaryCondition( bdCond, dirNodes, dirVals, neuNodes );
end
