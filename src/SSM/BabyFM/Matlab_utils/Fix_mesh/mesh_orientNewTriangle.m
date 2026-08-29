function s = mesh_orientNewTriangle( newFACE, myMesh, varargin )
%
% s = mesh_orientNewTriangle( newFACE, myMesh )
% s = mesh_orientNewTriangle( newFACE, myMesh, vertFaces, vertFaces_N )
% 
% Return the orientation of the triangle relative to the one of the mesh
% 1:     ok as it is
% -1:    must be inverted
% 0:     ambiguous (i.e. no contact with mesh edges)
%
% inf:   impossible to insert because of non-manifold geometry
% -inf:  cannot be inserted in the mesh with consistant orientation
% 

if isempty( varargin )
    [vertFaces, vertFaces_N] = mesh_vertexFaces( myMesh );
else
    if length( varargin ) ~= 2
        error('Incorrect number of inputs');
    else
        vertFaces = varargin{1};
        vertFaces_N = varargin{2};
    end
end

% Check signs of each edge
se = [...
    mesh_orientNewEdge( newFACE(1:2), myMesh, vertFaces, vertFaces_N ),...
    mesh_orientNewEdge( newFACE(2:3), myMesh, vertFaces, vertFaces_N ),...
    mesh_orientNewEdge( newFACE(3:-2:1), myMesh, vertFaces, vertFaces_N )];

% If any infinite, infinite
if sum( isfinite( se )) < 3    
    s = sum( se );
else    
    num_p = sum( se == 1 );
    if num_p == 3
        s = 1;
    else
        num_n = sum( se == -1 );
        if num_n == 3
            s = -1;
        else
            if num_p > 0 && num_n > 0
                s = -inf;
            else
                if sum( se == 0 ) == 3
                    s = 0;
                else
                    % Arrived here, recall that num_p and num_n cannot be
                    % both greater than zero
                    if num_p > 0
                        s = 1;
                    else 
                        s = -1;
                    end
                end
            end
        end
    end
end
