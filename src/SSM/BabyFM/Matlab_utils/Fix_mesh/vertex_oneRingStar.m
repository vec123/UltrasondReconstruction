function [vStar, neighbF, neighbV] = vertex_oneRingStar ( ...
    jv, myMesh, is_boundary_v, vertFaces, vertFaces_N )
%
% is_boundary_v:  is in reallity a guess used to speed-up the computation
% for vertices that are not in the boundary
%

neighbF = vertFaces( jv, 1 : vertFaces_N(jv));
if isempty( neighbF )
    vStar = struct(...
    'isSingular', {},...    
    'faceGroups',{});

    vStar(1).isSingular = 1;
    vStar(1).isBoundary = 0;
    vStar(1).nonManifoldEdges = 0;
    vStar(1).isDisconnected = 1;
    return;
end

% Here we assign numbers to groups of connected faces (i.e. faces that
% share an edge are connected)
f_connected = 0 * neighbF;

% We can start with an arbitrary face, except if this is a boundary vertex
% Hence we need to identify a boundary edge to start from
start_F = neighbF( 1 );
all_v_jF = myMesh.faces(:, start_F);
all_v_jF( all_v_jF == jv ) = [];
jv2 = min( all_v_jF );

if is_boundary_v
    neighbV = unique( myMesh.faces(:, neighbF) );
    neighbV( neighbV == jv ) = [];
    for jv2_idx = 1 : length( neighbV )
        jv2_t = neighbV( jv2_idx );        
        %neighbF1F2 = intersect( ...
        %    neighbF, vertFaces( jv2_t, 1 : vertFaces_N(jv2_t)) );        
        neighbF1F2 = neighbF( ismembc( neighbF, ...
            vertFaces( jv2_t, 1 : vertFaces_N(jv2_t))));
        if length( neighbF1F2 ) == 1
            jv2 = jv2_t;
            start_F = neighbF1F2;
            break;
        end
    end   

end
    
f_connected( neighbF == start_F ) = 1;
closed_star = false;
valence_jv = 0;
vStar = struct(...
    'isSingular', {},...    
    'faceGroups',{});

vStar(1).isSingular = 0;
vStar(1).isBoundary = 0;
vStar(1).nonManifoldEdges = 0;

% % To see what's up
% mesh_displayVstar( jv, neighbF, myMesh, vertFaces, vertFaces_N, 1 )

jF = start_F;
while not( closed_star )
    valence_jv = valence_jv + 1;    
    Tverts = myMesh.faces( :, jF );
    
    % Now select the edge that contains jv but not jv2
    Tverts( Tverts == jv ) = [];
    Tverts( Tverts == jv2 ) = [];
    jv3 = Tverts;
    
    % There are two faces sharing this edge
    neighbF3 = vertFaces( jv3, 1 : vertFaces_N(jv3));
    %neighbF1F3 = intersect( neighbF, neighbF3 );
    neighbF1F3 = neighbF( ismembc( neighbF, neighbF3));
    
    
    % We want the one that is not jF
    neighbF1F3( neighbF1F3 == jF ) = [];
    if length( neighbF1F3 ) > 1
        vStar.nonManifoldEdges = 1;
    end
    
    % Now there are two options:
    % - neighbF1F3 = start_F => back to the start:
    % -- if we visited all faces => the vertex is regular
    % -- if we didn't visit all faces => the vertex is singular
    % - neighbF1F3 is empty => then two more options
    % -- if we visited all faces => boundary vertex
    % -- if we didn't visit all faces => singular vertex
    
    if isempty( neighbF1F3 )
        if valence_jv == length( neighbF )
            vStar.isBoundary = 1;  
            not_visitedF = neighbF( f_connected == 0 );
            if isempty( not_visitedF )
                closed_star = 1;
            else
                [start_F, jv2] = vertex_OneRingDetermineNextFace (...
                    jv, neighbF, not_visitedF, myMesh, vertFaces, vertFaces_N);
                
                % Careful to determine an appropriate starting triangle
                cc_number = f_connected( neighbF == jF );
                f_connected( neighbF == start_F ) = cc_number + 1;
                jF = start_F;
                
                Tverts = myMesh.faces( :, start_F );
                Tverts( Tverts == jv ) = [];
                % Determine the best jv2
                jv2_a = Tverts(1);
                jv2_b = Tverts(2);
                %neighbF_12a = intersect( neighbF, vertFaces( jv2_a, 1 : vertFaces_N(jv2_a)) );
                neighbF_12a = neighbF( ismembc( neighbF, ...
                    vertFaces( jv2_a, 1 : vertFaces_N(jv2_a)) ));
                if length( neighbF_12a ) > 1
                    jv2 = jv2_b;
                else
                    % We dnn't need to check jv2_b because anyway jv2_a will not help
                    % neighbF_12b = intersect( neighbF, vertFaces( jv2_b, 1 : vertFaces_N(jv2_b)) );
                    jv2 = jv2_a;
                end
                
            end
        else
            vStar(1).isSingular = 1;            
                        
            % Set-up a new group to search for            
            not_visitedF = neighbF( f_connected == 0 );
            if isempty( not_visitedF )
                display( f_connected );
                error('cannot be empty here');
            end
            
            [start_F, jv2] = vertex_OneRingDetermineNextFace (...
                jv, neighbF, not_visitedF, myMesh, vertFaces, vertFaces_N);

            % Careful to determine an appropriate starting triangle
            cc_number = f_connected( neighbF == jF );
            f_connected( neighbF == start_F ) = cc_number + 1;
            jF = start_F;

            Tverts = myMesh.faces( :, start_F );
            Tverts( Tverts == jv ) = [];
            % Determine the best jv2
            jv2_a = Tverts(1);
            jv2_b = Tverts(2);
            %neighbF_12a = intersect( neighbF, vertFaces( jv2_a, 1 : vertFaces_N(jv2_a)) );
            neighbF_12a = neighbF( ismembc( neighbF, ...
                vertFaces( jv2_a, 1 : vertFaces_N(jv2_a))));
            
            
            if length( neighbF_12a ) > 1
                jv2 = jv2_b;
            else
                % We dnn't need to check jv2_b because anyway jv2_a will not help
                % neighbF_12b = intersect( neighbF, vertFaces( jv2_b, 1 : vertFaces_N(jv2_b)) );
                jv2 = jv2_a;
            end            

        end        
    else
        if neighbF1F3 == start_F
            if not( valence_jv == length( neighbF ))
                vStar(1).isSingular = 1;
            end                        
            
            not_visitedF = neighbF( f_connected == 0 );
            if isempty( not_visitedF )
                closed_star = 1; % OK, star is closed and complete
            else
                % Star is closed but there are extra triangles
                [start_F, jv2] = vertex_OneRingDetermineNextFace (...
                    jv, neighbF, not_visitedF, myMesh, vertFaces, vertFaces_N);

                % Careful to determine an appropriate starting triangle
                cc_number = f_connected( neighbF == jF );
                f_connected( neighbF == start_F ) = cc_number + 1;
                jF = start_F;

                Tverts = myMesh.faces( :, start_F );
                Tverts( Tverts == jv ) = [];
                % Determine the best jv2
                jv2_a = Tverts(1);
                jv2_b = Tverts(2);
                %neighbF_12a = intersect( neighbF, vertFaces( jv2_a, 1 : vertFaces_N(jv2_a)) );
                neighbF_12a = neighbF( ismembc( neighbF, ...
                    vertFaces( jv2_a, 1 : vertFaces_N(jv2_a))));
                if length( neighbF_12a ) > 1
                    jv2 = jv2_b;
                else
                    % We dnn't need to check jv2_b because anyway jv2_a will not help
                    % neighbF_12b = intersect( neighbF, vertFaces( jv2_b, 1 : vertFaces_N(jv2_b)) );
                    jv2 = jv2_a;
                end
            end
            
        else
            cc_number = f_connected( neighbF == jF );
            while f_connected( find( neighbF == neighbF1F3(1) ) ) > 0
                neighbF1F3(1) = [];
                if isempty( neighbF1F3 )
                    break;
                end
            end
            
            if isempty( neighbF1F3 )
                % closed_star = 1;
                not_visitedF = neighbF( f_connected == 0 );
                
                if isempty( not_visitedF )
                    closed_star = 1;
                else
                    closed_star = 0;
                    [start_F, jv2] = vertex_OneRingDetermineNextFace (...
                        jv, neighbF, not_visitedF, myMesh, vertFaces, vertFaces_N);
                    
                    % Careful to determine an appropriate starting triangle
                    cc_number = f_connected( neighbF == jF );
                    f_connected( neighbF == start_F ) = cc_number + 1;
                    jF = start_F;
                    
                    Tverts = myMesh.faces( :, start_F );
                    Tverts( Tverts == jv ) = [];
                    % Determine the best jv2
                    jv2_a = Tverts(1);
                    jv2_b = Tverts(2);
                    %neighbF_12a = intersect( neighbF, vertFaces( jv2_a, 1 : vertFaces_N(jv2_a)) );
                    neighbF_12a = neighbF( ismembc( neighbF, ...
                        vertFaces( jv2_a, 1 : vertFaces_N(jv2_a))));
                    if length( neighbF_12a ) > 1
                        jv2 = jv2_b;
                    else
                        % We dnn't need to check jv2_b because anyway jv2_a will not help
                        % neighbF_12b = intersect( neighbF, vertFaces( jv2_b, 1 : vertFaces_N(jv2_b)) );
                        jv2 = jv2_a;
                    end
                    
                end               
                
            else
                jF = neighbF1F3(1);  % The reference to index 1 is to cope with the case of non-manifold edges
                f_connected( neighbF == jF ) = cc_number;
                jv2 = jv3;
            end
        end
    end
    
end

% Verify validity of f_connected
if not( isempty( find( f_connected == 0, 1 )))
    error('Could not process all vertices in the star!');
end

%num_groups = length( unique( f_connected ));
num_groups = length( unique_vecFast( f_connected(:) ));

if not( isempty( find( f_connected > num_groups, 1 )))
    error('Non-consecutive group numbering');
end

if vStar.isSingular
    vStar.isBoundary = 1;
end
vStar.faceGroups = f_connected;

        
end




function [start_F, jv2] = vertex_OneRingDetermineNextFace (...
    jv, neighbF, not_visitedF, myMesh, vertFaces, vertFaces_N)

% Careful to determine an appropriate starting triangle
start_F = not_visitedF(1);
Tverts = myMesh.faces( :, start_F );
Tverts( Tverts == jv ) = [];
jv2 = Tverts(1);

not_visV = unique( myMesh.faces(:, not_visitedF) );
not_visV( not_visV == jv ) = [];
for jv2_idx = 1 : length( not_visV )
    jv2_t = not_visV( jv2_idx );
    %neighbF1F2 = intersect( ...
    %    neighbF, vertFaces( jv2_t, 1 : vertFaces_N(jv2_t)) );
    neighbF1F2 = neighbF( ismembc( neighbF, ...
        vertFaces( jv2_t, 1 : vertFaces_N(jv2_t)) ));
    if length( neighbF1F2 ) == 1
        if not( isempty( not_visitedF == neighbF1F2 ))
            jv2 = jv2_t;
            start_F = neighbF1F2;
            break;
        end
    end
end

end