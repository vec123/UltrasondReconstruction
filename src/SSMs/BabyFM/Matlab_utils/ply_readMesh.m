function myMesh = ply_readMesh2 ( Path )

% myMesh = ply_readMesh ( fileName )
% 
% Wrapper using PLY_READ but converting the output into a struct with
% fields verts, faces
% 

triData = ply_read( Path );
myMesh.verts = [triData.vertex.x, triData.vertex.y, triData.vertex.z]';
if isfield(triData.face, 'vertex_index')
    myMesh.faces = (cell2mat( triData.face.vertex_index ) + 1)';
else
    
    t2 = find( cellArray_getSubLenghts( triData.face.vertex_indices ) < 3 );
    if not( isempty( t2 ))
        for jt = 1 : length( t2 )
            old_t = triData.face.vertex_indices{ t2(jt) };
            triData.face.vertex_indices{ t2(jt) } = [old_t, old_t(1)];                
        end        
    end
    try
        myMesh.faces = (cell2mat( triData.face.vertex_indices ) + 1)';
    catch
        poly_N = cellArray_getSubLenghts( triData.face.vertex_indices );
%         warning('All input polygons (%d-%d pts) will be converted to triangles\n',...
%             min( poly_N ) : max( poly_N ));
        if min( poly_N ) < 3 || max( poly_N ) > 4
            error('Unsupported polygon type');
        end
        
        idxsT3 = find( poly_N == 3 );
        idxsT4 = find( poly_N == 4 );
        myMesh.faces = zeros(3, length( idxsT3 ) + ...
            2 * length( idxsT4 ));
        
        myMesh.faces(:, 1 : length( idxsT3 )) = ...
            (cell2mat( triData.face.vertex_indices( idxsT3 ) ) + 1)';
        polys4 = (cell2mat( triData.face.vertex_indices( idxsT4 ) ) + 1)';
        myMesh.faces(:, 1 + length( idxsT3 ) :...
            length( idxsT3 ) + length( idxsT4 )) = polys4( [1:3], : );
        myMesh.faces(:, 1 + length( idxsT3 ) + length( idxsT4 ) :...
            length( idxsT3 ) + 2 * length( idxsT4 )) = polys4( [1 3 4], : );
    end
end