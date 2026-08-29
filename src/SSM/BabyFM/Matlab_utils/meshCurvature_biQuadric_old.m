function [H, K, notEN, nNeigs] = meshCurvature_biQuadric_old( ...
    myMesh, neigb_radius, varargin)
%
% [H, K, notEN] = meshCurvature_biQuadric( myMesh, neigb_radius)
% (notEN = not-enough-neighbors, a vercotr with the indices of vertices not
% computed becuase there were less than 6 neighbors)
%
% Compute the curvature on a triangulated mesh by fitting a biquadric to each
% point using a neighorhood of radius 'neigb_radius'
%
% The implementation is based on Colombo et al (2006),
% Pattern Recognition 39 (2006) 444 – 455
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

if use_kd_tree
    fprintf(1, '\tBuilding kdtree...');
    myKdTree = kdtree_build( myMesh.verts' );
    fprintf(1, '\n');
% else    
%     fprintf(1, '\tComputing edges...');
%     [theEdgeLengths, theNeighbors, allL] = getMeshEdgeLenghts ( myMesh );
%     fprintf(1, '\tComputing neighborhoods...');
%     [nn_Neig, nn_Dist] = mesh_neigbWithinR (...
%         neigb_radius, theEdgeLengths, theNeighbors);
%     fprintf(1, '\n');
end

% Initialize curvatures to zero
NV = size( myMesh.verts, 2);
H = zeros( NV, 1);
K = zeros( NV, 1);
notEN = H;
nNeigs = H;

% fprintf(1, '\tProcesing vertices');
if verbose, fprintf(1, '\tProcesing vertices -> %3i%%',0); end
for jV = 1 : NV
%     if mod( jV, ceil(NV / 40) ) == 0
%         fprintf(1, '.');
%     end
    if verbose, fprintf('\b\b\b\b%3i%%',round((jV/NV)*100)); end
    
    if use_kd_tree
        v_idxs = kdtree_ball_query(...
            myKdTree, myMesh.verts(:, jV), neigb_radius);
    else
       
        dd_v = myMesh.verts - repmat(myMesh.verts(:,jV), [1 NV]);
        v_idxs = find( sum( dd_v.^2, 1 ) <= neigb_radius^2);
        % v_idxs = nn_Neig{jV};
    end

    nNeigs(jV) = length( v_idxs );
    if length( v_idxs ) < 6
        % Cannot compute the curvature with less than 6 neighbors
        % as the biquadric solves a linear system of rank 6
        notEN(jV) = 1;
    else
       
        % Define a biquadric in a neighborhood of (x0, y0)
        % s(x,y) = a + b*(x-x0) + c*(y-y0) + d*(x-x0)*(y-y0) + ...
        %    + e*(x-x0)^2 + f(y-y0)^2
        %
        % LINSOLVES: solve linear system A*X=B.
        % [1 difX difY difXY difYY difXX] * Matrix_abcdef = SURF_VALUES
        %        
       
        x0 = myMesh.verts(1, jV);
        y0 = myMesh.verts(2, jV);
        z0 = myMesh.verts(3, jV);
       
        AA = ones(length(v_idxs), 6);
        
%         BB = myMesh.verts(3, v_idxs)';
%        
%         patchX = myMesh.verts(1, v_idxs) - x0;
%         patchY = myMesh.verts(2, v_idxs) - y0;
%                
%         AA(:, 1) = 1;
%         AA(:, 2) = patchX;
%         AA(:, 3) = patchY;
%         AA(:, 4) = patchX .* patchY;
%         AA(:, 5) = patchX .^ 2;
%         AA(:, 6) = patchY .^ 2;
% ------------ Reprojection on a local coordinate system        
        % Compute PCA axes of the 3D point cloud
        [U,~,~]= pca_Santi( myMesh.verts(:, v_idxs) );
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

            % Now the partial derivatives are
            % fX = b     fXX = 2e     fXY = d
            % fY = c     fYY = 2f
            fX = abcdef(2);
            fY = abcdef(3);
            fXY = abcdef(4);
            fXX = 2 * abcdef(5);
            fYY = 2 * abcdef(6);        

            %
            % And the curvatures can be computed as (always from Colombo 2006)
            %
            %      (1 + fY^2) * fXX - 2 * fX * fY * fXY  + (1 + fX^2) * fYY
            % H = ------------------------------------------------------------
            %                  2 * (1 + fX^2 + fY^2)^(3/2)
            %
            %       fXX * fYY - fXY ^ 2        
            % K = -----------------------
            %      ((1 + fX^2 + fY^2)^2      
            %

            H(jV) = (...
                (1 + fY^2) * fXX - 2 * fX * fY * fXY + (1 + fX^2) * fYY...
                ) / (2 * (1 + fX^2 + fY^2)^(3/2));
            K(jV) = ...
                (fXX * fYY - fXY ^ 2) /...
                ((1 + fX^2 + fY^2)^2);
%
%             % Using I and II fundamental forms
%             I_E = 1 + fX^2;
%             I_F = fX * fY;
%             I_G = 2 + fY^2;
%             II_L = fXX;
%             II_M = fXY;
%             II_N = fYY;
%            
%             H(jV) = (II_L * I_G - 2 * II_M * I_F + II_N * I_E) / ...
%                 (2 * I_E * I_G - I_F^2);
%             K(jV) = (II_L * II_N - II_M^2) / (I_E * I_G - I_F^2);
           
%             if abs( H(jV) - old_H ) > 1e-10
%                 warning('Disagreement in H');
%             end
%             if abs( K(jV) - old_K ) > 1e-10
%                 warning('Disagreement in K');
%             end    
           
           
        end % IF rank ok (6)    
    end % IF enough elements (>=6)
end % jV

% fprintf(1,' \n');
if use_kd_tree
    kdtree_delete( myKdTree );
end
