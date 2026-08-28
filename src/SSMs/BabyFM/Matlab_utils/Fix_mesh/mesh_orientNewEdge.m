function s = mesh_orientNewEdge( newEDGE, myMesh, vertFaces, vertFaces_N )
%
% s = mesh_orientNewEdge( newEDGE, myMesh, vertFaces, vertFaces_N )
% 
% Return the orientation of a new edge relative to the one of the mesh
% 1:    ok as it is
% -1:   must be inverted
% 0:    ambiguous (i.e. no contact with mesh edges)
% inf:  should not be added: will be non-manifold
%

neighbF_v1 = vertFaces( newEDGE(1), 1 : vertFaces_N( newEDGE(1) ));
neighbF_v2 = vertFaces( newEDGE(2), 1 : vertFaces_N( newEDGE(2) ));
i_v1v2 = intersect( neighbF_v1, neighbF_v2 );

if isempty( i_v1v2 )
    s = 0;
else
    if length( i_v1v2 ) == 1
        s = - triangEdge_inducedSign( ...
            myMesh.faces(:, i_v1v2), newEDGE );
    else
        s = inf;
    end
end



