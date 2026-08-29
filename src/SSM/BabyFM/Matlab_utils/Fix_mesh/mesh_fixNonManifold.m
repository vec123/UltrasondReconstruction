function [out_mesh, infoSTR, meshChanges] = mesh_fixNonManifold( myMesh, varargin )
%
% This function is under development.
% So far, it does:
% - Removal of degenerate triangles
% - Removal of non-manifold edges
%
% Syntax:
%   [out_mesh, infoSTR] = mesh_fixNonManifold( myMesh [,options] )
%
% Options 
%   'nofix'         ->   Doesn't fix, just reports   
%   'noduplic'      ->   Disable vertex duplication. Some singular vertices
%                        will not be removed (DEFAULT)
%   'allowDuplic'   ->   Enable vertex duplication. Some singular vertices
%                        will not be removed
%   'silent'        ->   No messages
%   'verbose'       ->   Show step-by-step messages
%   'display'       ->   Display each fix (and pause)
%
%   'avoidAdd'      ->   First option for fixes is removal, latest is
%                        additions (default)
%   'avoidRemove'   ->   First option for fixes is addition, latest is
%                         removal 
%


% Defaults
fix_problems = true;
verbose_level = 1;
show_plots = 0;
allow_duplic = false;
avoid_add = true;


% Var-options
while not( isempty( varargin ))
    valid_option = false;
    
    if isempty( varargin{1} )
        valid_option = true;
    end
    
    if strcmpi( varargin{1}, 'nofix' )
        valid_option = true;
        fix_problems = false;
    end
    
    if strcmpi( varargin{1}, 'silent' )
        valid_option = true;
        verbose_level = 0;
    end
    
    if strcmpi( varargin{1}, 'verbose' )
        valid_option = true;
        verbose_level = 2;
    end    
    
    if strcmpi( varargin{1}, 'display' )
        valid_option = true;
        show_plots = 1;
    end   
    
    if strcmpi( varargin{1}, 'noduplic' )
        valid_option = true;
        allow_duplic = false;
    end   
    
    if strcmpi( varargin{1}, 'allowDuplic' )
        valid_option = true;
        allow_duplic = true;
    end       
        
    if strcmpi( varargin{1}, 'avoidAdd' )
        valid_option = true;
        avoid_add = true;
    end   
     
    if strcmpi( varargin{1}, 'avoidRemove' )
        valid_option = true;
        avoid_add = false;
    end  
        
    if not( valid_option )
        error('Unrecognized input opton: %s', varargin{1});
    end
    varargin(1) = [];   
end

fake3D_from_2D = false;
if size( myMesh.verts, 1 ) == 2
    myMesh.verts = [myMesh.verts; zeros( 1, size( myMesh.verts, 2 ))];
    fake3D_from_2D = true;
else
    if size( myMesh.verts, 1 ) ~= 3
        error('Input meshes must be in 2D or 3D');
    end
end


infoSTR = struct();
infoSTR(1).singularVerts = struct;
infoSTR(1).singularVerts(1).number = 0;
infoSTR(1).singularVerts(1).fixed_by_case = 0;
infoSTR(1).singularVerts(1).fixed_wRestart = 0;
infoSTR(1).singularVerts(1).fixed_by_duplic = 0;
infoSTR(1).singularVerts(1).not_fixed = 0;

infoSTR(1).nonManifoldEdges = struct;
infoSTR(1).nonManifoldEdges(1).number = 0;
infoSTR(1).nonManifoldEdges(1).fixed = 0;

re_started_list = [];

% Record all changes performed to the mesh so that they can be reproduced
meshChanges = struct();
meshChanges(1).op = 'Start';
meshChanges(1).info = {'nVerts', size( myMesh.verts, 2 ),...
    'nFaces', size( myMesh.faces, 2 )};

[myMesh, zaFaces] = mesh_removeZeroAreaFaces( myMesh, 1e-10, 'silent' );
if not( isempty( zaFaces ))
    meshChanges( length( meshChanges ) + 1 ).op =...
        'mesh_RemoveFaces';
    meshChanges( end ).param = zaFaces;
    meshChanges( end ).info = 'Remove zero-area face(s)';
end

[myMesh, rFaces] = mesh_removeInvalidFaces( myMesh, 'silent' );
if not( isempty( rFaces ))
    meshChanges( length( meshChanges ) + 1 ).op =...
        'mesh_RemoveFaces';
    meshChanges( end ).param = rFaces;
    meshChanges( end ).info = 'Remove invalid face(s)';
end

[vertFaces, vertFaces_N] = mesh_vertexFaces( myMesh );

% Remove disconnected vertices
nc_verts = find( vertFaces_N == 0 );
if not( isempty( nc_verts )) 
    if verbose_level > 0
        fprintf(1, '\n*** Found %d disconnected vertices...',...
            length( nc_verts ));
    end
    
    meshChanges( length( meshChanges ) + 1 ).op =...
        'mesh_RemoveDisconnectedVerts';
    meshChanges( end ).param = nc_verts;
    
    [myMesh, not_connected] = mesh_RemoveDisconnectedVerts( myMesh, 'silent' );        
    if max( abs( sort( not_connected(:) ) - sort( nc_verts(:) ))) > 0
        error('Inconsistency in disconnected vertices');
    end
    
    [vertFaces, vertFaces_N] = mesh_vertexFaces( myMesh );
    if not( isempty( find( vertFaces_N == 0, 1 )))
        error('Unexpected disconnected vertices after attempting removal');
    end
    
    if verbose_level > 0
        fprintf(1, '\n');
    end
end

% Initialize
NV = size( myMesh.verts, 2 );%no of vertices
isBoundary = zeros(1, NV);
inWrongEdge = zeros(1, NV);
isSingular = zeros(1, NV);

if verbose_level > 0
    fprintf ('Processing %d vertices ... ', NV);
    if verbose_level == 1
        fprintf ('%6d', 0);
    end
end

jv = 1;
while jv <= size( myMesh.verts, 2 )    
    % Find the faces sorrounding the vertex    
    neighbF = vertFaces( jv, 1 : vertFaces_N(jv));    
    
    % Get the neighboring vertices, and remove current one
    %neighbV = unique( myMesh.faces(:, neighbF) );
    neig_faces = myMesh.faces(:, neighbF);    
    neighbV = unique_vecFast( neig_faces(:) );
    
    neighbV( neighbV == jv ) = [];         
%     if jv == 19927
%         jv
%     end
       
    % Check for wrong edges
    % --------------------------------------
    % For every edge connecting this vertex we check the number of
    % simplices involved: if only one then it is part of the boundary
    for jv2_idx = 1 : length( neighbV )
        jv2 = neighbV( jv2_idx );
        if inWrongEdge(jv) > 0 && inWrongEdge(jv2) > 0
            continue;
        end
        try
            neighbF2 = vertFaces( jv2, 1 : vertFaces_N(jv2));
        catch
            warning('Temporary try-catch');
            myMesh = mesh_RemoveDisconnectedVerts( myMesh );
            break;
        end

        % check the number of faces containing the edge
        % neighbF1F2 = intersect( neighbF, neighbF2 );        
        neighbF1F2 = neighbF( ismembc( neighbF, neighbF2));
        
        if length( neighbF1F2 ) > 2
            inWrongEdge(jv) = length( neighbF1F2 );
            infoSTR(1).nonManifoldEdges(1).number = ...
                infoSTR(1).nonManifoldEdges(1).number + 1;

            if not( fix_problems )
                % If we don't fix it then we mark jv2 as well
                % Otherwise we would count this edge twice
                inWrongEdge(jv2) = inWrongEdge(jv);
            else
                [myMesh, vertFaces, vertFaces_N, fixedOK, meshChanges] = ...
                    mesh_fixNonManifoldEdge( jv, jv2, myMesh, meshChanges, ...
                    vertFaces, vertFaces_N, verbose_level, show_plots);

                if not( fixedOK )                                    
                    infoSTR(1).nonManifoldEdges(1).fixed = ...
                        infoSTR(1).nonManifoldEdges(1).fixed + 1;
                    jv = 0;
                    isBoundary = 0 * isBoundary;
                    inWrongEdge = 0 * isBoundary;
                    isSingular = 0 * isBoundary;
                    if verbose_level > 0
                        fprintf(1,...
                            '\n\tRE-STARTING after NM removal w/sing-v:       ');
                    end
                    break;
                else            
                    infoSTR(1).nonManifoldEdges(1).fixed = ...
                        infoSTR(1).nonManifoldEdges(1).fixed + 1;
                    isBoundary = [isBoundary, 0];
                    inWrongEdge = [inWrongEdge, 0];
                    isSingular = [isSingular, 0];
                    continue;
                end
                
            end
        else
            if length( neighbF1F2 ) == 1 
                % It is safe to set the boundary flag ON
                isBoundary(jv) = 1;                
            end
        end
    end   
    
    if jv > 0

        % If the edges are OK we can still have a singular vertex
        % For this, starting at an arbitrary face we should be able to close
        % the 1-ring star of vertex jv
        if inWrongEdge(jv) == 0
            vStar = vertex_oneRingStar ( ...
                jv, myMesh, isBoundary(jv), vertFaces, vertFaces_N ); 

            if verbose_level > 0
                if not( isBoundary(jv) == vStar.isBoundary )
                    fprintf(1, 'WARNING: Inconsistent boundary identification\n');
                end
            end

            isSingular( jv ) = vStar.isSingular;        
            if vStar.isSingular
                infoSTR(1).singularVerts(1).number = ...
                    infoSTR(1).singularVerts(1).number + 1;

                if fix_problems                
                    [myMesh, vertFaces, vertFaces_N, fixed_v, fix_attempt, meshChanges] = ...
                        mesh_fixSingularVertex( jv, myMesh, meshChanges, vertFaces, vertFaces_N,...
                        verbose_level, show_plots, allow_duplic, avoid_add);

                    if fix_attempt > 1
                        if length( find( re_started_list == jv )) < 3
                            re_started_list = [re_started_list, jv];                        
                            infoSTR(1).singularVerts(1).fixed_wRestart =...
                                    infoSTR(1).singularVerts(1).fixed_wRestart + 1;                            
                            jv = 1;
                            isBoundary = 0 * isBoundary;
                            inWrongEdge = 0 * isBoundary;
                            isSingular = 0 * isBoundary;
                            if verbose_level > 0
                                fprintf(1,...
                                    '\n\tRE-STARTING after SV removal w/artifacts:       ');
                            end

                            continue;
                        else
                            if verbose_level > 0
                                fprintf(1,...
                                    '\n\t***UNABLE TO FIX - SKIPPING VERTEX:       ');
                                infoSTR(1).singularVerts(1).not_fixed =...
                                    infoSTR(1).singularVerts(1).not_fixed + 1;                                   
                            end                            
                        end
                    else
                        if fixed_v > 0
                            infoSTR(1).singularVerts(1).fixed_by_case =...
                                infoSTR(1).singularVerts(1).fixed_by_case + 1;
                        else
                            if allow_duplic
                                infoSTR(1).singularVerts(1).fixed_by_duplic =...
                                    infoSTR(1).singularVerts(1).fixed_by_duplic + 1;
                            end
                        end

                        inWrongEdge( jv ) = 0;
                        isBoundary( jv ) = 0;
                        if allow_duplic || fixed_v > 0
                            isSingular( jv ) = 0;
                        end

                        while size( myMesh.verts, 2 ) > NV
                            isBoundary = [isBoundary, 0];
                            inWrongEdge = [inWrongEdge, 0];
                            isSingular = [isSingular, 0];
                            NV = NV + 1;
                        end
                        
                        if allow_duplic || fixed_v > 0
                            continue;                        
                        else
                            if verbose_level > 1
                                fprintf(1, '\n\t\t*** SINGULAR VERTEX NOT REMOVED');
                            end
                        end
                    end
                end
            end

        end
    end
    
    jv = jv + 1;
    
    if verbose_level == 1
        if mod(jv, 1000) == 0
            fprintf ('\b\b\b\b\b\b%6d', jv);
        end
    end
end


if verbose_level > 0
    fprintf(1, '\n');
    fprintf(1, '\tNon-manifold edges = %d\n', ...
        infoSTR(1).nonManifoldEdges(1).number);
    fprintf(1, '\tSingular vertices = %d\n',...
        infoSTR(1).singularVerts(1).number);
end

if not( fix_problems )
    infoSTR(1).singularVerts(1).isSingular = isSingular;
    infoSTR(1).nonManifoldEdges(1).inWrongEdge = inWrongEdge;
end

out_mesh = myMesh;
if fake3D_from_2D
    out_mesh.verts = myMesh.verts(1:2, :);
end  
    
end


% ---------------------------------------------------------------------
% ---------------------------------------------------------------------
%                       NON-MANIFOLD EDGES
% ---------------------------------------------------------------------
% ---------------------------------------------------------------------
function [myMesh, vertFaces, vertFaces_N, edge_fixed, meshChanges] = ...
    mesh_fixNonManifoldEdge( jv, jv2, myMesh, meshChanges, vertFaces, vertFaces_N,...
    verbose_level, show_plots)
    
    if verbose_level > 1
        fprintf(1, '\n\tNON-MANIFOLD EDGE [%d-%d]...', ...
            jv, jv2);
    end
    
    neighbF = vertFaces( jv, 1 : vertFaces_N(jv));                
    neighbF2 = vertFaces( jv2, 1 : vertFaces_N(jv2));    
    %neighbF1F2 = intersect( neighbF, neighbF2 );
    neighbF1F2 = neighbF( ismembc( neighbF, neighbF2));
    
    % Check if any of the faces contain boundary vertices    
    all_nVerts = unique( myMesh.faces(:, neighbF1F2) );
    all_nVerts_isBoundary = zeros( length( all_nVerts ), 1 );
    for jAV = 1 : length( all_nVerts )        
        temp_vStar = vertex_oneRingStar ( ...
            all_nVerts(jAV), myMesh, 1, vertFaces, vertFaces_N );
        all_nVerts_isBoundary( jAV ) = temp_vStar.isBoundary;
    end
    
    f1f2_isBoundary = zeros( length( neighbF1F2 ), 1 );
    for jF12 = 1 : length( f1f2_isBoundary )
        Tverts = myMesh.faces(:, neighbF1F2( jF12 ));
        for j3 = 1 : length( Tverts )
            if all_nVerts_isBoundary( all_nVerts == Tverts( j3 ) )
                f1f2_isBoundary( jF12 ) = f1f2_isBoundary( jF12 ) + 1;
            end
        end
    end
    
    [nada, try_remove_idx] = sort( f1f2_isBoundary, 'descend' );    
    
    % Attempt to remve the face with highest boundary verts
    edge_fixed = false;
    for jAttempt = 1 : 2
        try_remove = neighbF1F2( try_remove_idx );
        while not( edge_fixed )
            if isempty( try_remove )
                break;
            end

            if verbose_level > 1
                fprintf(1, '\n\t\tAttempting removal of face %d...', try_remove(1));
            end
            f_remove = try_remove(1);
            try_remove( 1 ) = [];

            % Check all vertices of this triangles
            vertex_became_singular = 0;        

            % We check here that none becomes singular
            % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            Tverts = myMesh.faces( :, f_remove );
            tMesh = mesh_RemoveFaces( myMesh, f_remove );
            [t_vertFaces, t_vertFaces_N] = mesh_vertexFaces( tMesh );

            for j3 = 1 : 3
                vStar_j3 = vertex_oneRingStar ( ...
                    Tverts( j3 ), tMesh, 1, t_vertFaces, t_vertFaces_N );
                if vStar_j3.isSingular
                    vertex_became_singular = 1;
                end
            end

            disp_depth = 3;
            if show_plots
                clf; subplot(1, 2, 1);
                mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );            
            end

            % If all vertices are OK, then we continue
            if vertex_became_singular == 0 || jAttempt == 2
                if verbose_level > 1
                    if jAttempt == 1
                        fprintf(1, 'ok');
                    else
                        fprintf(1, 'Fixed but new sing verts');
                    end
                end
                if show_plots                    
                    subplot(1, 2, 2);
                    mesh_displayVstar( jv, neighbF, tMesh, vertFaces, vertFaces_N, disp_depth );
                    getframe; pause;
                    subplot(1, 2, 1);
                    g2 = Import_GCApositionSettings;
                    subplot(1, 2, 2);
                    set_GCApositionSettings( g2 );
                    getframe; pause;
                end
                
                t_neighbF = t_vertFaces( jv, 1 : t_vertFaces_N(jv));                
                t_neighbF2 = t_vertFaces( jv2, 1 : t_vertFaces_N(jv2));    
                %t_neighbF1F2 = intersect( t_neighbF, t_neighbF2 );
                t_neighbF1F2 = t_neighbF( ismembc( t_neighbF, t_neighbF2));

                if length( t_neighbF1F2 ) <= 2 || jAttempt == 2
                    % Accept the fix
                    myMesh = tMesh;
                    vertFaces = t_vertFaces;
                    vertFaces_N = t_vertFaces_N;
                    
                    % This comes from:
                    %   tMesh = mesh_RemoveFaces( myMesh, f_remove );
                    meshChanges( length( meshChanges ) + 1 ).op =...
                        'mesh_RemoveFaces';
                    meshChanges( end ).param = f_remove;
                    meshChanges( end ).info = 'Remove face at non-manifold edge';
                    
                    edge_fixed = true;
                end
            else
                if verbose_level > 1
                    fprintf(1, 'FAILED');
                end
            end

        end % WHILE NOT( edge_fixed )
        
        if edge_fixed
            % If jAttmpt == 2 then we didn't really fix the problem but
            % replaced by a new one, so the cleaning has to be restarted
            if jAttempt == 2
                edge_fixed = false;
            end
            break;
        end
        
    end % fOR jAttempt
    
    if verbose_level > 1
        if edge_fixed
            fprintf('\n\t\tfixed');
        else
            fprintf('\n\t\t*** FAILED TO FIX');
        end            
    end
end









