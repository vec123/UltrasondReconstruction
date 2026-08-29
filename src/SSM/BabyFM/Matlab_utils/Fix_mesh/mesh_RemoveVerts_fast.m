function outMesh = mesh_RemoveVerts_fast( mesh_3D, v_to_clean )
%
v_to_clean = sort(v_to_clean);
outMesh = mesh_3D;

% Find the faces to which vertices belong
[vertFaces, vertFaces_N] = mesh_vertexFaces(mesh_3D);
f_to_clean=unique( vertFaces(v_to_clean,:)); % Faces that we want to remove
non=(f_to_clean==-1); % Delete the -1
f_to_clean(non) = [];

outMesh.faces(:, f_to_clean ) = []; % Remove faces

v=zeros(length(mesh_3D.verts),1); 
v(v_to_clean)=1;
in=find(~double(v)==1); % vertices index that we keep
real_v = 1:length(in); % new vertices index
faces_n = reshape(outMesh.faces,[],1); % reshape to one column vector
[Lf Lv] = ismember(faces_n ,in); % find where and which index appear in each position

faces_n(Lf)= real_v(Lv); % Create new triangulation (replace the old vertex index by the new)
oput_f = reshape(faces_n,3,[]); % reshape to 3xN (faces)

outMesh.faces = oput_f; % Assing new triangulation


 % Remove the vertex coordinates
outMesh.verts( :, v_to_clean ) = [];



end