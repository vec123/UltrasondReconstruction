function [myNormals, notEN, nNeigs] = meshNormals_BiQuadric_new( ...
    myMesh, neigb_radius, varargin)
%
% [myNormals, notEN] = meshNormals_BiQuadric( myMesh, neigb_radius)
% [myNormals, notEN] = meshNormals_BiQuadric( myMesh, neigb_radius, 'no_kdtree')
%
% (notEN = not-enough-neighbors, a vector with the indices of vertices not
% computed becuase there were less than 8 neighbors)
%
% Compute the normals on a triangulated mesh by fitting a biquadric to each
% point using a neighorhood of radius 'neigb_radius'
% 


use_kd_tree = true;
verbose = 0;
while not( isempty( varargin ))
    if strcmpi( varargin{1}, 'no_kdtree' )
        use_kd_tree = false;
        varargin(1) = [];
        
        continue;
    end
    
    if strcmpi( varargin{1}, 'verbose' )
        verbose = 1;
        varargin(1) = [];
        
        continue;
    end
    
    error('Unrecognized input argument %s', varargin{1})
end

% Initialize curvatures to zero
NV = size( myMesh.verts, 2);
myNormals = zeros( 3, NV );

notEN = zeros( NV, 1 );
nNeigs = zeros( NV, 1 );


% Compute mesh centroid (to correct normal orientations)
MC = mean( myMesh.verts, 2 );

fprintf(1, '\n\tComputing neighbours ');
all_neig = rangesearch( myMesh.verts', myMesh.verts', neigb_radius);


% fprintf(1, '\tProcesing vertices ');
if verbose, fprintf(1, '\tProcesing vertices -> %3i%%',0); end
for jV = 1 : NV
%     if mod( jV, ceil(NV / 40) ) == 0
%         fprintf(1, '.');
%     end

    if verbose, fprintf('\b\b\b\b%3i%%',round((jV/NV)*100)); end

%     if use_kd_tree
%         v_idxs = kdtree_ball_query(...
%             myKdTree, myMesh.verts(:, jV), neigb_radius);
%     else        
%         dd_v = myMesh.verts - repmat(myMesh.verts(:,jV), [1 NV]);
%         v_idxs = find( sum( dd_v.^2, 1 ) <= neigb_radius^2);
%     end

    
    v_idxs = all_neig{jV};
    
    nNeigs(jV) = length( v_idxs );
    if length( v_idxs ) < 8
        % Cannot compute the curvature with less than 6 neighbors
        % as the biquadric solves a linear system of rank 6
        notEN(jV) = 1;
        
        % We compute this normal based on the face normals
        j_faces = find(...
            myMesh.faces(1,:) == jV |...
            myMesh.faces(2,:) == jV |...
            myMesh.faces(3,:) == jV );
        
        % If the faces are consistantly oriented, it doesn't matter the order of
        % vertices, that is for a face [v1,v2,v3]:
        %   cross(v2-v1, v3-v1)
        %   cross(v3-v2, v1-v3)
        %   cross(v1-v3, v2-v1)
        % produce all the same result
        face_normals = zeros(3, length( j_faces ));
        for j2 = 1 : length( j_faces )
            face_verts = myMesh.verts(:, myMesh.faces(:, j_faces(j2) ));
            face_normals(:, j2) = cross(...
                face_verts(:, 2) - face_verts(:, 1),...
                face_verts(:, 3) - face_verts(:, 1));
        end
        
        if length( j_faces ) > 1
            myNormals(:, jV) = mean( face_normals, 2 );
        else
            myNormals(:, jV) = face_normals;
        end
        
    else
        
        % Define a biquadric in a neighborhood of (x0, y0)
        % s(x,y) = a + b*(x-x0) + c*(y-y0) + d*(x-x0)*(y-y0) + ...
        %    + e*(x-x0)^2 + f(y-y0)^2
        %
        % LINSOLVES: olve linear system A*X=B.
        % [1 difX difY difXY difYY difXX] * Matrix_abcdef = SURF_VALUES
        %        
        
        x0 = myMesh.verts(1, jV);
        y0 = myMesh.verts(2, jV);
        z0 = myMesh.verts(3, jV);
        
        AA = ones(length(v_idxs), 6);
        
        %BB = myMesh.verts(3, v_idxs)';        
        %patchX = myMesh.verts(1, v_idxs) - x0;
        %patchY = myMesh.verts(2, v_idxs) - y0;
               
% ------------ Reprojection on a local coordinate system        
        % Compute PCA axes of the 3D point cloud
        [U,D,media]= pca_Santi( myMesh.verts(:, v_idxs) );
        ejeX = U(:,1);
        ejeY = U(:,2);
        ejeZ = U(:,3);
        
        % We have to project dX,dY,dZ into the new axes
        refX = myMesh.verts(1, v_idxs) - x0;
        refY = myMesh.verts(2, v_idxs) - y0;
        refZ = myMesh.verts(3, v_idxs) - z0;
        dX = refX * ejeX(1) + refY * ejeX(2) + refZ * ejeX(3);
        dY = refX * ejeY(1) + refY * ejeY(2) + refZ * ejeY(3);
        dZ = refX * ejeZ(1) + refY * ejeZ(2) + refZ * ejeZ(3);        
% ------------ END OF Reprojection on a local coordinate system

        AA(:, 1) = 1;
        AA(:, 2) = dX;
        AA(:, 3) = dY;
        AA(:, 4) = dX .* dY;
        AA(:, 5) = dX .^ 2;
        AA(:, 6) = dY .^ 2;
        BB = dZ';
        
        % Least squares estimation of the quadric parameters
        [abcdef, estim_rank] = linsolve(AA,BB);
        if estim_rank >= 6

            % Normal in the rotated space
			rotN = [abcdef(1:2); -1];
			
			% Project back into original space
            myNormals(1, jV) = rotN(1) * ejeX(1) + rotN(2) * ejeY(1) + rotN(3) * ejeZ(1);
            myNormals(2, jV) = rotN(1) * ejeX(2) + rotN(2) * ejeY(2) + rotN(3) * ejeZ(2);
            myNormals(3, jV) = rotN(1) * ejeX(3) + rotN(2) * ejeY(3) + rotN(3) * ejeZ(3);
           
        end % IF rank ok (6)    
    end % IF enough elements (>=6)
    
    % Normalize							
    myTwoNormSQ = sum( myNormals(:, jV) .^2 );
    if myTwoNormSQ >= 1e-10 
        myNormals(:, jV) = myNormals(:, jV) / sqrt( myTwoNormSQ );
    end            

    % Correct orientation
    d_jV_to_MC = myMesh.verts(:, jV) - MC;
    if dot( myNormals(:, jV), d_jV_to_MC ) < 0
        myNormals( :, jV ) = - myNormals( :, jV );
    end

end 

