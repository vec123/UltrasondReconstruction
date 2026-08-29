% ---------------------------------------------------------------------
% ---------------------------------------------------------------------
%                       SINGULAR VERTICES
% ---------------------------------------------------------------------
% ---------------------------------------------------------------------
% fix_type returns
%  0: if the vertex could not be fixed
% -1: if the vertex was fixed by duplication
% >0: if the vertex was fixed (indicates the case number used for the fix)
%     1: Add a triangle
%     11: Remove disconnected triangle
%     12: Remove triangle
%     13: Remove groups of triangles
%     
%
function [myMesh, vertFaces, vertFaces_N, fix_type, fix_attempt, meshChanges] = ...
    mesh_fixSingularVertex( jv, myMesh, meshChanges, vertFaces, vertFaces_N,...
    verbose_level, show_plots, allow_duplic, avoid_add)   

    fix_type = 0;
    if verbose_level > 1
        fprintf(1, '\n\tSingular vertex %d:', jv);
    end
    
    [vStar, neighbF, neighbV] = vertex_oneRingStar ( ...
        jv, myMesh, 1, vertFaces, vertFaces_N );
   
    disp_depth = 3;
    if show_plots
        clf; subplot(1, 2, 1);
        mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );
    end
     

    % fix_attempt = 1  -->  Proceed only if everything OK
    % fix_attempt = 2  -->  Proceed if jv gets ok despite other artifacts
    % fix_attempt = 3  -->  Remove something even if the problem remains
    fix_attempt = 0;
    
    while fix_attempt < 3
        if fix_attempt < 1
            if not( avoid_add )
                fix_attempt = fix_attempt + 0.5;
            else
                fix_attempt = 1;
            end
        else
            fix_attempt = fix_attempt + 1;
        end
        
        % Case 1: Search for disconnected triangles
        % --------------------------------------------------------
        if fix_attempt >= 1
            nGroups = max( vStar.faceGroups );
            jg = 0;
            while jg < nGroups
                jg = jg + 1;
                G_idxs = find( vStar.faceGroups == jg );
                if length( G_idxs ) == 1
                    % We analyze this isolated face of the star
                    jF = neighbF( G_idxs );
                    Tverts = myMesh.faces( :, jF );
                    Tverts( Tverts == jv ) = [];
                    neighbF2 = vertFaces( Tverts(1), 1 : vertFaces_N( Tverts(1) ));
                    neighbF3 = vertFaces( Tverts(2), 1 : vertFaces_N( Tverts(2) ));

                    % If the other two verts neighbor only this face, the triangle
                    % is isolated from the mesh
                    nF2F3 = intersect( neighbF2, neighbF3 );
                    if length( nF2F3 ) == 1
                        if nF2F3 == jF
                            
                            % There is still a problem here: if the face to
                            % remove is the only one that contains jv, then
                            % we cannot do this fix (i.e. this triangle
                            % should be removed from the vertex that is
                            % touching the rest of the mesh)
                            [ijk, fff] = find( myMesh.faces == jv );
                            fff( fff == jF ) = [];
                            if not( isempty( fff ))

                                if verbose_level > 1
                                    fprintf(1, '\n\t\tRemoving face %d', jF);
                                end

                                [myMesh, vertChanges] = mesh_RemoveFaces( myMesh, jF );                            
                                meshChanges( length( meshChanges ) + 1 ).op =...
                                    'mesh_RemoveFaces';
                                meshChanges( end ).param = jF;
                                meshChanges( end ).info = 'Remove disconnected triangle';
                                
                                % We have to check if jv has changed index
                                % due to the removal of jF
                                if not( isempty( vertChanges ))
                                    jvCHANGES = find( vertChanges(:,1) == jv );
                                    if not( isempty( jvCHANGES ))
                                        if length( jvCHANGES ) > 1
                                            error('Multiple changes of jv - This should not happen');
                                        end
                                        jv = vertChanges(jvCHANGES, 2);
                                    end
                                end
                                
                                [vertFaces, vertFaces_N] = mesh_vertexFaces( myMesh );       
                                try
                                    [vStar, neighbF, neighbV] = vertex_oneRingStar ( ...
                                        jv, myMesh, 1, vertFaces, vertFaces_N );
                                catch
                                    error('Unexpected internal error');
                                end
                                if vStar.isSingular == 0
                                    if verbose_level > 1
                                        fprintf(1, '\n\t\tok');                        
                                    end
                                    if show_plots
                                        subplot(1, 2, 2);
                                        mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );
                                        getframe; pause;                            
                                        subplot(1, 2, 1);
                                        g2 = Import_GCApositionSettings;
                                        subplot(1, 2, 2);
                                        set_GCApositionSettings( g2 );
                                        getframe; pause;
                                    end
                                    fix_type = 11;
                                    return;
                                end
                                nGroups = max( vStar.faceGroups );
                                jg = jg - 1;
                            end
                        end
                    end

                end   
            end
        end
        
        % Case 1-b: Try removing a triangle that, 
        % although is not isolated in the mesh, is isolated in the v-star
        % ---------------------------------------------------------------
        if fix_attempt >= 1
            nGroups = max( vStar.faceGroups );
            jg = 0;
            while jg < nGroups
                jg = jg + 1;
                G_idxs = find( vStar.faceGroups == jg );
                if length( G_idxs ) == 1
                    % We analyze this isolated face of the star
                    jF = neighbF( G_idxs );

                    % If the other two verts neighbor only this face, the triangle
                    % is isolated from the mesh - 
                    % We know this is not the case because that was accounted for
                    % in case 1. We try removing it anyway and check if any of the
                    % attached vertices becomes singular            
                    if verbose_level > 1
                        fprintf(1, '\n\t\tAttempting removal of face(s) ');
                        for jj = 1 : length( jF )
                            fprintf(1, '%d ', jF(jj));
                        end
                    end

                    % Check if the vertices becaome singular
                    Tverts = myMesh.faces(:, jF);
                    Tverts = unique( Tverts(:) );
                    Tverts( Tverts == jv ) = [];
                    
                    [ijk, fff] = find( myMesh.faces == jv );
                    fff( fff == jF ) = [];
                    if isempty( fff )
                        continue;
                    end
                    
                    [tMesh, vertChanges] = mesh_RemoveFaces( myMesh, jF );   
                    [t_vertFaces, t_vertFaces_N] = mesh_vertexFaces( tMesh );

                    not_feasible = 0;
                    if fix_attempt < 2
                        for jj = 1 : length( Tverts )
                            vStar_2 = vertex_oneRingStar ( ...
                                Tverts( jj ), myMesh, 1, vertFaces, vertFaces_N );
                            vStar_2NEW = vertex_oneRingStar ( ...
                                Tverts( jj ), tMesh, 1, t_vertFaces, t_vertFaces_N );

                            if (vStar_2NEW.isSingular > vStar_2.isSingular)  
                                if verbose_level > 1
                                    fprintf(1, ' - FAILED: New singular vertices');
                                end
                                not_feasible = 1;
                                break;
                            end
                        end
                    end

                    if not_feasible == 0
                        if verbose_level > 1
                            fprintf(1, ' Ok');
                        end
                        old_mesh = myMesh;

                        % Validate the fix
                        myMesh = tMesh;                    
                        vertFaces = t_vertFaces;
                        vertFaces_N = t_vertFaces_N;

                        meshChanges( length( meshChanges ) + 1 ).op =...
                            'mesh_RemoveFaces';
                        meshChanges( end ).param = jF;
                        meshChanges( end ).info = sprintf(...
                            'Remove 1-face group from 1-ring of V %d', jv);

                        [vStar, neighbF, neighbV] = vertex_oneRingStar ( ...
                            jv, myMesh, 1, vertFaces, vertFaces_N );
                        if vStar.isSingular == 0 || fix_attempt > 2
                            if verbose_level > 1
                                fprintf(1, '\n\t\tok');
                            end
                            if show_plots
                                subplot(1, 2, 2);
                                mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );
                                getframe; pause;
                                subplot(1, 2, 1);
                                hold on;
                                old_faces = old_mesh.faces(:, jF);
                                old_X = old_mesh.verts(1, old_faces);
                                old_Y = old_mesh.verts(2, old_faces);
                                old_Z = old_mesh.verts(3, old_faces);
                                patch( old_X, old_Y, old_Z, 'r' );
                                camlight headlights;

                                g2 = Import_GCApositionSettings;
                                subplot(1, 2, 2);
                                set_GCApositionSettings( g2 );
                                getframe; pause;
                            end             

                            fix_type = 12;
                            return;
                        end
                        nGroups = max( vStar.faceGroups );
                        jg = jg - 1;                
                    end               

                end   
            end
        end


        % Case 2: There is just a missing triangle
        % --------------------------------------------------------
        % Find out whether two boundary vertices from different face-groups
        % could be connected without causing non-manifold triangles
        nGroups = max( vStar.faceGroups );
        jg = 0;
        while jg + 1 < nGroups
            jg = jg + 1;
            jg2 = jg + 1;
            G1_idxs = find( vStar.faceGroups == jg );
            G2_idxs = find( vStar.faceGroups == jg2 );
            for j_G1 = 1 : length( G1_idxs )
                Fg1 = neighbF( G1_idxs( j_G1 ));
                verts_Fg1 = myMesh.faces( :, Fg1 );
                verts_Fg1( verts_Fg1 == jv ) = [];

                for j_G2 = 1 : length( G2_idxs )                
                    Fg2 = neighbF( G2_idxs( j_G2 ));                               
                    verts_Fg2 = myMesh.faces( :, Fg2 );                                
                    verts_Fg2( verts_Fg2 == jv ) = [];

                    for k_v1_idx = 1 : length( verts_Fg1 )
                        k_v1 = verts_Fg1( k_v1_idx );

                        % Only boundary vets are acceptable candidates
                        aux_star = vertex_oneRingStar ( ...
                            k_v1, myMesh, 1, vertFaces, vertFaces_N );
                        if aux_star.isBoundary == 0
                            continue;
                        end

                        for k_v2_idx = 1 : length( verts_Fg2 )
                            k_v2 = verts_Fg2( k_v2_idx );

                            % Only boundary vets are acceptable candidates
                            aux_star = vertex_oneRingStar ( ...
                                k_v2, myMesh, 1, vertFaces, vertFaces_N );
                            if aux_star.isBoundary == 0
                                continue;
                            end

                            neighbF_k1 = vertFaces( k_v1, 1 : vertFaces_N( k_v1 ));
                            neighbF_k2 = vertFaces( k_v2, 1 : vertFaces_N( k_v2 ));
                            i_k1k2 = intersect( neighbF_k1, neighbF_k2 );
                            if length( i_k1k2 ) == 1
                                % The edge exists in another triangle, so
                                % it could be added, but we need to see if
                                % that would generate non-manifold edges
                                % This would be the case if any of the
                                % edges in the new triangle already
                                % neighbor two faces
                                i_k1jv = intersect( neighbF_k1, neighbF );
                                i_k2jv = intersect( neighbF_k2, neighbF );

                                if max([length(i_k1jv), length(i_k2jv)]) < 2
                                    if verbose_level > 1
                                        fprintf(1,...
                                            '\n\t\tAdding triangle from existing edges');
                                    end

                                    % Ok, add the new triangle (with the
                                    % appropriate sign)
                                    ind_sign_1 = triangEdge_inducedSign( ...
                                        myMesh.faces(:, i_k1k2), [k_v1, k_v2] );
                                    ind_sign_2 = triangEdge_inducedSign( ...
                                        myMesh.faces(:, i_k1jv), [jv, k_v1] );
                                    ind_sign_3 = triangEdge_inducedSign( ...
                                        myMesh.faces(:, i_k2jv), [k_v2, jv] );

                                    if (ind_sign_1 == ind_sign_2) && (ind_sign_1 == ind_sign_3)
                                        if ind_sign_1 > 0
                                            newFace = [jv, k_v2, k_v1]';
                                        else
                                            newFace = [jv, k_v1, k_v2]';
                                        end
                                        
                                        myMesh.faces = [myMesh.faces, newFace];
                                        meshChanges( length( meshChanges ) + 1 ).op =...
                                            'addTriangles';
                                        meshChanges( end ).param = newFace;
                                        meshChanges( end ).info = sprintf(...
                                            'Add triangle at singular V %d', jv);                                        
                                        
                                        newNF = size( myMesh.faces, 2 );
                                        vertFaces( jv, vertFaces_N(jv) + 1 ) = ...
                                            newNF;
                                        vertFaces( k_v1, vertFaces_N(k_v1) + 1 ) = ...
                                            newNF;
                                        vertFaces( k_v2, vertFaces_N(k_v2) + 1 ) = ...
                                            newNF;
                                        vertFaces_N( jv ) = vertFaces_N( jv ) + 1;
                                        vertFaces_N( k_v1 ) = vertFaces_N( k_v1 ) + 1;
                                        vertFaces_N( k_v2 ) = vertFaces_N( k_v2 ) + 1;

                                        % Update star and diminish jg
                                        [vStar, neighbF, neighbV] = vertex_oneRingStar ( ...
                                            jv, myMesh, 1, vertFaces, vertFaces_N );
                                        if vStar.isSingular == 0
                                            if verbose_level > 1
                                                fprintf(1, '\n\t\tok');
                                            end
                                            if show_plots
                                                subplot(1, 2, 2);
                                                mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );
                                                getframe; pause;                                        
                                                subplot(1, 2, 1);
                                                g2 = Import_GCApositionSettings;
                                                subplot(1, 2, 2);
                                                set_GCApositionSettings( g2 );
                                                getframe; pause;
                                            end       

                                            fix_type = 1;
                                            return;
                                        end

                                        nGroups = max( vStar.faceGroups );
                                        jg = jg - 1;

                                        % No subsequent tests in current setting
                                        k_v1_idx = inf;
                                        k_v2_idx = inf;
                                        j_G1 = inf;
                                        j_G2 = inf;

                                    else
                                        if verbose_level > 1
                                            fprintf(1, '\n\t\tFAILED (unorientable)');
                                        end                                    
                                    end


                                end
                            end
                        end
                    end                                                            
                end            
            end   
        end
        
        
        % Case 2-b: There is just a missing triangle but we need to add an
        % edge as well
        % ----------------------------------------------------------------
        % Find out whether two boundary vertices from different face-groups
        % could be connected without causing non-manifold triangles
        nGroups = max( vStar.faceGroups );
        jg = 0;
        merit_str = struct();
        merit_cost = [];
        while jg + 1 < nGroups
            jg = jg + 1;
            jg2 = jg + 1;
            G1_idxs = find( vStar.faceGroups == jg );
            G2_idxs = find( vStar.faceGroups == jg2 );
            
            for j_G1 = 1 : length( G1_idxs )
                Fg1 = neighbF( G1_idxs( j_G1 ));
                verts_Fg1 = myMesh.faces( :, Fg1 );
                verts_Fg1( verts_Fg1 == jv ) = [];

                for j_G2 = 1 : length( G2_idxs )                
                    Fg2 = neighbF( G2_idxs( j_G2 ));                               
                    verts_Fg2 = myMesh.faces( :, Fg2 );                                
                    verts_Fg2( verts_Fg2 == jv ) = [];

                    for k_v1_idx = 1 : length( verts_Fg1 )
                        k_v1 = verts_Fg1( k_v1_idx );

                        % Only boundary vets are acceptable candidates
                        aux_star = vertex_oneRingStar ( ...
                            k_v1, myMesh, 1, vertFaces, vertFaces_N );
                        if aux_star.isBoundary == 0
                            continue;
                        end

                        for k_v2_idx = 1 : length( verts_Fg2 )
                            k_v2 = verts_Fg2( k_v2_idx );

                            % Only boundary vets are acceptable candidates
                            aux_star = vertex_oneRingStar ( ...
                                k_v2, myMesh, 1, vertFaces, vertFaces_N );
                            if aux_star.isBoundary == 0
                                continue;
                            end

                            merit_cost = [merit_cost, -sum( ...
                                ( myMesh.verts(:, k_v1) -...
                                myMesh.verts(:, k_v2) ).^2 )];
                            merit_str( length( merit_cost ) ).kv = ...
                                [k_v1, k_v2];                                                        
                        end
                    end                                                            
                end            
            end   
        end        
        
        if not( isempty( merit_cost ))
            % If any feasible connection, test it
            [sort_merit, sort_jC] = sort( merit_cost, 'descend' );
            for jC_idx = 1 : length( merit_cost )
                jC = sort_jC( jC_idx );
                k_v1 = merit_str(jC).kv(1);            
                k_v2 = merit_str(jC).kv(2);

                % Check the sign of the new triangle
                newFACE = [k_v1, k_v2, jv]';
                s = mesh_orientNewTriangle( newFACE, myMesh, vertFaces, vertFaces_N );
                if not( s == 1 )
                    if s == -1
                        newFACE = [k_v2, k_v1, jv]';
                    else
%                         if verbose_level > 1
%                             fprintf(1, '\n\t\tFAILED (unorientable)');
%                         end
                        % Cannot accept this one, go on to next
                        continue;
                    end
                end

                % Test this inclusion
                tMesh = myMesh;
                tMesh.faces = [tMesh.faces, newFACE(:)];
                [t_vertFaces, t_vertFaces_N] = ...
                    mesh_vertexFaces( tMesh );

                aux_star1 = vertex_oneRingStar ( ...
                    k_v1, tMesh, 1, t_vertFaces, t_vertFaces_N );
                if not( aux_star1.isSingular )
                    aux_star2 = vertex_oneRingStar ( ...
                        k_v2, tMesh, 1, t_vertFaces, t_vertFaces_N );
                    if not( aux_star2.isSingular )
                        aux_star3 = vertex_oneRingStar ( ...
                            jv, tMesh, 1, t_vertFaces, t_vertFaces_N );
                        if not( aux_star3.isSingular )
                            % OK, successful
                            if verbose_level > 1
                                fprintf(1,...
                                    '\n\t\tAdding edge from existing verts');
                            end

                            myMesh = tMesh;
                            vertFaces = t_vertFaces;
                            vertFaces_N = t_vertFaces_N;

                            meshChanges( length( meshChanges ) + 1 ).op =...
                                'addTriangles';
                            meshChanges( end ).param = newFACE;
                            meshChanges( end ).info = sprintf(...
                                'Add triangle at singular V %d', jv);

                            % Update star and diminish jg
                            [vStar, neighbF, neighbV] = vertex_oneRingStar ( ...
                                jv, myMesh, 1, vertFaces, vertFaces_N );
                            if vStar.isSingular == 0
                                if verbose_level > 1
                                    fprintf(1, '\n\t\tok');
                                end
                                if show_plots
                                    subplot(1, 2, 2);
                                    mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );
                                    getframe; pause;
                                    subplot(1, 2, 1);
                                    g2 = Import_GCApositionSettings;
                                    subplot(1, 2, 2);
                                    set_GCApositionSettings( g2 );
                                    getframe; pause;
                                end

                                fix_type = 2;
                                return;
                            else
                                error(1, 'INTERNAL ERROR');
                            end
                        end
                    end        
                end
            end
        end
    
    
        % Case 3: Try removing a group of triangles that, 
        % although is not isolated in the mesh, is isolated in the v-star
        % --------------------------------------------------------------
        % Try 2-element up to 3-element groups
        if fix_attempt >= 1
            for nf_out = 2 : 2 + fix_attempt

                nGroups = max( vStar.faceGroups );
                jg = 0;
                while jg < nGroups
                    jg = jg + 1;
                    G_idxs = find( vStar.faceGroups == jg );
                    if length( G_idxs ) == nf_out
                        % We analyze this isolated group of the star
                        jF = neighbF( G_idxs );

                        % If the other two verts neighbor only this face, the triangle
                        % is isolated from the mesh - 
                        % We know this is not the case because that was accounted for
                        % in case 1. We try removing it anyway and check if any of the
                        % attached vertices becomes singular            
                        if verbose_level > 1
                            fprintf(1, '\n\t\tAttempting removal of face(s) ');
                            for jj = 1 : length( jF )
                                fprintf(1, '%d ', jF(jj));
                            end
                        end

                        % Check if the vertices becaome singular
                        Tverts = myMesh.faces(:, jF);
                        Tverts = unique( Tverts(:) );
                        Tverts( Tverts == jv ) = [];
                        [ijk, fff] = find( myMesh.faces == jv );                                                
                        for j7 = 1 : length( jF )
                            fff( fff == jF(j7) ) = [];
                        end
                        if isempty( fff )
                            continue;
                        end
                                                   
                        tMesh = mesh_RemoveFaces( myMesh, jF );   
                        [t_vertFaces, t_vertFaces_N] = mesh_vertexFaces( tMesh );

                        not_feasible = 0;
                        if fix_attempt < 2
                            for jj = 1 : length( Tverts )
                                if Tverts( jj ) > length( t_vertFaces_N )
                                    continue;
                                end
                                
                                vStar_2 = vertex_oneRingStar ( ...
                                    Tverts( jj ), myMesh, 1, vertFaces, vertFaces_N );
                                vStar_2NEW = vertex_oneRingStar ( ...
                                    Tverts( jj ), tMesh, 1, t_vertFaces, t_vertFaces_N );
                                
                                if (vStar_2NEW.isSingular > vStar_2.isSingular)
                                    if verbose_level > 1
                                        fprintf(1, ' - FAILED: New singular vertices');
                                    end
                                    not_feasible = 1;
                                    break;
                                end
                            end
                        end

                        if not_feasible == 0
                            if verbose_level > 1
                                fprintf(1, ' Ok');
                            end

                            % Validate the fix
                            myMesh = tMesh;
                            vertFaces = t_vertFaces;
                            vertFaces_N = t_vertFaces_N;

                            meshChanges( length( meshChanges ) + 1 ).op =...
                                'mesh_RemoveFaces';
                            meshChanges( end ).param = jF;
                            meshChanges( end ).info = sprintf(...
                                'Remove %d-faces group from the 1-ring of V %d', jv);

                            [vStar, neighbF, neighbV] = vertex_oneRingStar ( ...
                                jv, myMesh, 1, vertFaces, vertFaces_N );
                            if vStar.isSingular == 0 || fix_attempt > 2
                                if verbose_level > 1
                                    fprintf(1, '\n\t\tok');
                                end
                                if show_plots
                                    subplot(1, 2, 2);
                                    mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );
                                    getframe; pause;
                                    subplot(1, 2, 1);
                                    g2 = Import_GCApositionSettings;
                                    subplot(1, 2, 2);
                                    set_GCApositionSettings( g2 );
                                    getframe; pause;
                                end             

                                fix_type = 13;
                                return;
                            end
                            nGroups = max( vStar.faceGroups );
                            jg = jg - 1;                
                        end               

                    end   
                end
            end
        end

        % If nothing worked, try duplicating the vertex
        % --------------------------------------------------------------
        if vStar.isSingular == 1

            if verbose_level > 1
                fprintf(1, '\n\t\t*** Unable to resolve with the current connectivity');
            end

            if allow_duplic

                if verbose_level > 1
                    fprintf(1, '\n\t\t*** Generating duplicate indices for this vertex...');        
                end

                num_of_stars = length( unique( vStar.faceGroups ));
                for jG = 2 : num_of_stars                    
                    myMesh.verts = [myMesh.verts, myMesh.verts(:, jv)];
                    meshChanges( length( meshChanges ) + 1 ).op =...
                        'addVerteices';
                    meshChanges( end ).param = myMesh.verts(:, jv);
                    meshChanges( end ).info = sprintf(...
                        'Duplicating vertex V %d', jv);

                    new_v_idx = size( myMesh.verts, 2 );
                    f_idxs = neighbF( vStar.faceGroups == jG );
                    old_faces = myMesh.faces( :, f_idxs );
                    new_faces = old_faces;
                    new_faces( old_faces == jv ) = new_v_idx;
                    
                    myMesh.faces( :, f_idxs ) = new_faces;
                    meshChanges( length( meshChanges ) + 1 ).op =...
                        'ReplaceTriangles';
                    meshChanges( end ).param = {'f_idxs', f_idxs,...
                        'new_faces', new_faces};
                    meshChanges( end ).info = sprintf(...
                        'Duplicating vertex V %d', jv);
                end

                [vertFaces, vertFaces_N] = mesh_vertexFaces( myMesh );
                [vStar, neighbF, neighbV] = vertex_oneRingStar ( ...
                    jv, myMesh, 1, vertFaces, vertFaces_N );
                if vStar.isSingular == 0
                    if verbose_level > 1
                        fprintf(1, '\n\t\tok');
                    end
                    if show_plots
                        subplot(1, 2, 2);
                        mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );
                        getframe; pause;
                        subplot(1, 2, 1);
                        g2 = Import_GCApositionSettings;
                        subplot(1, 2, 2);
                        set_GCApositionSettings( g2 );
                        getframe; pause;
                    end                  

                    fix_type = -1;
                    return;
                end

                if vStar.isSingular == 1
                    fix_type = 0;
                    if verbose_level > 1
                        fprintf(1, '\n\t\t************************************');
                        fprintf(1, '\n\t\t*** NOT ABLE TO RESOLVE SINGULARITY');
                        fprintf(1, '\n\t\t************************************');                            
                    end
                    disp_depth = 4;
                    if show_plots
                        clf; subplot(1, 2, 1);
                        mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, disp_depth );
                        getframe;
                        pause;
                    end                        
                    error('');
                end    
            end
        end
    
    end
    
end

