function [m_res, allLengths] =...
    mesh_ComputeResolution ( myMesh )

% [theEdgeLengths, theNeighbors, allLengths] = ...
%       getMeshEdgeLenghts ( myMesh );
% 
% Computes the lengths of edges and the neighbors to each vertex
% NOTE: The vertors are sorted by NUMBER OF VERTEX, in the same
% order as the VERTS matrix
%
% While theEdgeLengths and theNeighbors are cell arrays, allLengths
% is just a row vector with all lengths (to compute average lenght 
% etc)/
%

m_faces = myMesh.faces;
m_verts = myMesh.verts;

if (size( m_verts, 1 ) ~= 3) || ...
    (size( m_faces, 1 ) ~= 3)
    m_verts = m_verts';
    m_faces = m_faces';
    if (size( m_verts, 1 ) ~= 3) || ...
        (size( m_faces, 1 ) ~= 3)
        error('Triangulated mesh with verts,faces fields expected');
    else
        fprintf(1, '\nWARNING: verts and faces are transposed w.r.t. expected');
    end
end

NF = size( m_faces, 2 );
edgeLength2 = zeros( NF * 3, 1 );
if NF > 1e5
    fprintf ('Procesing %d faces ==> %7d', NF, 0);
end

je = 0;
for jf = 1 : NF
    
    vv = m_faces(:, jf);
    edgeLength2( je+1: je+3) = ...
        sum(( m_verts(:, vv) - m_verts(:, vv([2 3 1]) )).^2);
    je = je + 3;
       
    if mod(jf, 10000) == 0
        if NF > 1e5
            fprintf ('\b\b\b\b\b\b\b%7d', NF, jf);
        end
    end
end


allLengths = sqrt( edgeLength2 );
m_res = median( allLengths );

fprintf( '\n');



    