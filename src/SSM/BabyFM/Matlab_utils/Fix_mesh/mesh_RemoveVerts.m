function outMesh = mesh_RemoveVerts( mesh_3D, v_to_clean )
%
v_to_clean = sort(v_to_clean);
outMesh = mesh_3D;
rFaces = zeros( size( mesh_3D.faces, 2 ), 1);
for kv_idx = 1 : length( v_to_clean )
    kv = v_to_clean( kv_idx );
    [~, new_f] = find( mesh_3D.faces == kv );
    rFaces( new_f ) = 1;
end
f_to_clean = find( rFaces == 1 );
outMesh.faces(:, f_to_clean ) = [];

for jvc = 1 : length( v_to_clean )
    % Remove the vertex coordinates
    v_to_remove = v_to_clean( jvc );
    outMesh.verts( :, v_to_remove ) = [];
    
    % Downgrade all indices above it
    outMesh.faces( outMesh.faces > v_to_remove ) = outMesh.faces( outMesh.faces > v_to_remove ) - 1;
    v_to_clean = v_to_clean - 1;
end