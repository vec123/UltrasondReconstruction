function [mesh_out, invalidF] = mesh_removeZeroAreaFaces( mesh_in, zeroAreaThresh, varargin )
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
zeroAreaThresh_SQ_2 = 2 * zeroAreaThresh * zeroAreaThresh;

for jf = 1 : NF
    Tverts = mesh_in.faces(:, jf);
    Tpoints = mesh_in.verts(:, Tverts);
    
    % SLOW ALTERNATIVE
    %if triang3D_Area( Tpoints ) < zeroAreaThresh
    %    valid_faces(jf) = 0;
    %end
    
    a = Tpoints(:,1) - Tpoints(:,2);
    b = Tpoints(:,1) - Tpoints(:,3);
    tArea_SQ_2 = (...
        ( a(2,:).*b(3,:)-a(3,:).*b(2,:) )^2 +...
        ( a(3,:).*b(1,:)-a(1,:).*b(3,:) )^2 +...
        ( a(1,:).*b(2,:)-a(2,:).*b(1,:) )^2 );
    if tArea_SQ_2 < zeroAreaThresh_SQ_2
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

