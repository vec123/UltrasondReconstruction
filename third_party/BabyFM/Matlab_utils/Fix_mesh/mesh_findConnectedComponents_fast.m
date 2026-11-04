function [v_groups, outMesh, g_count] = ...
    mesh_findConnectedComponents_fast( myMesh, varargin )
% 
% [v_groups, myMesh, g_count] = mesh_findConnectedComponents( myMesh )
% [v_groups, myMesh] = mesh_findConnectedComponents( myMesh, 'clean' )
%
% v_groups:  is a vector indicating a group number foe each vertex  
%            Group numbers are consecutive starting at 1
% outMesh:   is the mesh after cleaning (only if 'clean' is specified or if
%            disconnected vertices are identified)
% g_count:   indicates the number of vertices in each group
%
% 'clean'    is an optional argument that instructs the function to remove
%            all vertices that do not belong to the majority group
%

do_cleaning = 0;
have_neigs = 0;
verbose_level = 1;

while not( isempty( varargin ))
    if strcmpi( varargin{1}, 'clean' )
        do_cleaning = 1;
        varargin(1) = [];
        continue;
    end
    
    if strcmpi( varargin{1}, 'haveNeigs' )
        have_neigs = 1;
        vertFaces = varargin{2};
        vertFaces_N = varargin{3};
        varargin(1:3) = [];
        continue;
    end    
    
    if strcmpi( varargin{1}, 'silent' )
        verbose_level = 0;
        varargin(1) = [];
        continue;
    end

    if strcmpi( varargin{1}, 'verbose' )
        verbose_level = 2;
        varargin(1) = [];
        continue;
    end    
    
    error('Unrecognized input argument: %s', varargin{1});        
end

if have_neigs
    outMesh = myMesh;
else
    outMesh = mesh_removeInvalidFaces( myMesh );
    [vertFaces, vertFaces_N] = mesh_vertexFaces( outMesh );    
end

% Remove disconnected vertices
nc_verts = find( vertFaces_N == 0 );
if not( isempty( nc_verts ))    
    if verbose_level > 0
        fprintf(1, '\nRemoving %d disconnected vertices...',...
            length( nc_verts ));
        fprintf(1, '\n*** THIS MEANS THE RETURNED INDICES DO NOT MATCH');
        fprintf(1, '\n*** THOS ON THE ORIGINAL MESH - USE THE ONE RETURNED INSTEAD\n');
    end
    
        nc_verts = sort( nc_verts, 'ascend' );
    
        % Remove the vertex coordinates
        
        outMesh.verts( :, nc_verts ) = [];
        vertFaces( nc_verts, : ) = [];
        vertFaces_N( nc_verts, : ) = [];

        
        % Downgrade vertexs 
        v=zeros(length(myMesh.verts),1);
        v(nc_verts)=1;
        in=find(~double(v)==1);  % vertices index that we keep
        real_v = 1:length(in);  % new vertices index
        faces_n = reshape(outMesh.faces,[],1); % reshape to one column vector
        [Lf Lv] = ismember(faces_n ,in); % find where and which index appear in each position

        faces_n(Lf)= real_v(Lv); % Create new triangulation (replace the old vertex index by the new)
        oput_f = reshape(faces_n,3,[]);  % reshape to 3xN (faces)
        
        outMesh.faces= oput_f; % Assing new triangulation


%     NO NEED FOR THIS BECAUSE WE REMOVED ISOLATED VERTICES
%     [vertFaces2, vertFaces_N2] = mesh_vertexFaces( outMesh );
%     if not( isempty( find( vertFaces_N2 == 0, 1 )))
%         error('Unexpected disconnected vertices after attempting removal');
%     end
    
    if verbose_level > 0
        fprintf(1, '\n');
    end
end

% Identify connected components of the mesh
v_groups = zeros(1, size( outMesh.verts, 2 ));

while not( isempty( find( v_groups == 0, 1 )))
    % Select a non-grouped vertex
    jv = find( v_groups == 0, 1 );
    
    % Get its neighbors
    neighbF = vertFaces( jv, 1 : vertFaces_N(jv));        
    neighbV = unique( outMesh.faces(:, neighbF) );
    
    % If any of its neighbors is non-zero, group with them
    g_neigs_nonzero_idx = find( v_groups( neighbV ) > 0 );
    g_neigs_nonzero = neighbV( g_neigs_nonzero_idx );
    if not( isempty( g_neigs_nonzero ))
        % This should not actually happen
        v_groups( jv ) = min( v_groups( g_neigs_nonzero ) );        
        g_uids = unique( v_groups( g_neigs_nonzero ) );
        for j2 = 1 : length( g_uids )
            neig_group_id = g_uids( j2 );
            if neig_group_id == v_groups( jv )
                continue;
            else
                % All neighbors with other numbers actually belong to this
                % same group
                v_groups( v_groups == neig_group_id ) = v_groups( jv );
            end
        end              
    else
        v_groups( jv ) = max( v_groups ) + 1;        
    end        
    
    % Of course, all neighbors are connected to the same group    
    v_groups( neighbV ) = v_groups( jv );
end

% Now we have to check if there are not inter-group intersections
% ---------------------------------------------------------------
allG = sort( unique( v_groups ) );
for j1_idx = 2 : length( allG )
    % Group to test
    g = allG( j1_idx );
    
    % Find vertices in this group
    g_verts = find( v_groups == g );
    
    if isempty( g_verts )
        continue;
    end
    
    % Find all neighbors to g_verts
    g_neigF = unique( vertFaces( g_verts, : ));
    g_neigF( g_neigF == -1 ) = [];
    g_neigV = unique( outMesh.faces(:, g_neigF) );
    
    % Check if these vertices belong to any group but 'g'
    g_neigs_Groups = unique( v_groups( g_neigV ));
    g_neigs_Groups( g_neigs_Groups == g ) = [];
    if not( isempty( g_neigs_Groups ))
        for j2_idx = 1 : length( g_neigs_Groups )
            g2 = g_neigs_Groups( j2_idx );
            v_groups( v_groups == g ) = g2;
        end
    end
end
        
allG = sort( unique( v_groups ) );
if verbose_level > 0
    fprintf(1, '\nNumber of components = %d\n', length( allG ));
end

g_count = zeros(1, length( allG ));
for jg = 1 : length( allG )
    if not( allG( jg ) == jg )
        if allG( jg ) < jg
            error('Unexpected group number');
        end
        v_groups( v_groups == allG( jg ) ) = jg;
    end
    g_count( jg ) = length( find( v_groups == jg ));
end

if not( max( v_groups(:) ) == length( allG ))
    warning('Group numbering might be inconsistent\n');    
end

if length( allG ) > 1
    if do_cleaning
        [nada, g_main] = max( g_count );        
        v_to_clean = sort( find( v_groups ~= g_main ), 'ascend');
        f_to_clean = unique( vertFaces( v_to_clean, : ));
        f_to_clean( f_to_clean == -1 ) = [];
        outMesh.faces(:, f_to_clean ) = [];
        
        if verbose_level > 0
            fprintf(1, '\tRemoving secondary components (%d verts, %d faces)...',...
                length( v_to_clean ), length( f_to_clean ));
        end
        
       
    
        
        % Downgrade vertexs 
        v=zeros(length(outMesh.verts),1);
        v( v_to_clean)=1;
        in=find(~double(v)==1); % vertices index that we keep
        real_v = 1:length(in); % new vertices index
        faces_n = reshape(outMesh.faces,[],1); % reshape to one column vector
        [Lf Lv] = ismember(faces_n ,in); % find where each vertex appear

        faces_n(Lf)= real_v(Lv); % Create new triangulation (replace the old vertex index by the new)
        
        oput_f = reshape(faces_n,3,[]); % reshape to 3xN (faces)
        outMesh.faces= oput_f; % Assing new triangulation
        
        % Remove the vertex coordinates
        
        outMesh.verts( :, v_to_clean ) = [];
        vertFaces( v_to_clean, : ) = [];
        vertFaces_N( v_to_clean, : ) = [];
          
        
        if verbose_level > 0
            fprintf(1, '\n');              
        end
    end
end

