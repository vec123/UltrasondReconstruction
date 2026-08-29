function [mesh_out, invalidF] = mesh_removeInvalidFaces( mesh_in, varargin )
% Remove invalid triangles (zero-area)

be_silent = 0;
if not( isempty( varargin ))
    if strcmpi( varargin{1}, 'silent' )
        be_silent = 1;
    else
        error('Unrecognized parameter');
    end
end

NF = size( mesh_in.faces, 2 );
valid_faces = ones(NF, 1);
for jf = 1 : NF
    temp_f = mesh_in.faces(:, jf);
    if length( unique_vecFast( temp_f(:) ) ) < 3
        valid_faces(jf) = 0;
    end
end

if sum( valid_faces ) < NF
    if not( be_silent )
        fprintf(1, '%d zero-area triangles removed\n',...
            NF - sum( valid_faces ));
    end
    invalidF = find( valid_faces == 0 );
else
    invalidF = [];
end

mesh_out.verts = mesh_in.verts;
mesh_out.faces = mesh_in.faces( :, valid_faces == 1 );

