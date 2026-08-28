function [vertFaces, vertFaces_N] = mesh_vertexFaces( myMesh )
% vertFaces, vertFaces_N]
%
% Find the faces to which vertices belong
% * vertFaces is a matrix, one row per vertex, indicating the faces
% adjacent to each vertex
% * vertFaces_N is a vector that indicates the number of faces per vertex
% (after vertFaces_N(jv) the vertex jv has "-1" entries in vertFaces
%
% Because "unique" is used, each row of vertFaces is sorted
%

NV = size( myMesh.verts, 2);
big_vertFaces = -ones( NV, 100);
vertFaces_N = zeros(NV, 1);

for jf = 1 : size (myMesh.faces, 2)   
    vertFaces_N( myMesh.faces(:, jf) ) = ...
        vertFaces_N( myMesh.faces(:, jf) ) + 1;
    for jv_idx = 1 : 3
        big_vertFaces( myMesh.faces(jv_idx, jf), ...
            vertFaces_N( myMesh.faces(jv_idx, jf))) = jf;
    end     
end

% Make unique
max_num_faces = 0;
for jv = 1 : NV
    % unique_verts = unique( big_vertFaces(jv, 1:vertFaces_N(jv)));
    if vertFaces_N(jv) > 1
        unique_verts = unique_vecFast( big_vertFaces(jv, 1:vertFaces_N(jv))' )';
        big_vertFaces( jv, 1:length(unique_verts) ) = unique_verts;
        big_vertFaces( jv, length(unique_verts)+1:end) = -1;
        vertFaces_N(jv) = length(unique_verts);
        if length( unique_verts ) > max_num_faces
            max_num_faces = length( unique_verts );
        end
    end
end

vertFaces = big_vertFaces(:,[1 : max_num_faces]);
    

