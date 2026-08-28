function [newMesh, vertChanges] = mesh_RemoveFaces (myMesh, a0)

vertChanges = [];
newMesh = myMesh;
newMesh.faces( :, a0 ) = [];

a0_verts = myMesh.faces(:, a0);
a0_verts = sort( unique( a0_verts(:) ), 'descend' );

for jv_idx = 1 : length( a0_verts )
    jv = a0_verts( jv_idx );
    if isempty( find( newMesh.faces == jv, 1) )
        if jv < size( newMesh.verts, 2 )
            newMesh.verts(:, jv) =  newMesh.verts(:, end);
            vertChanges = [vertChanges; [size( newMesh.verts, 2 ), jv]]; 
            newMesh.faces( newMesh.faces == size( newMesh.verts, 2 ) ) = jv;            
        end
        newMesh.verts(:, end) = [];
    end
end


% One problem still remains: multiple changes of indices. For example:
%        36201       36197
%        36200       36186
%        36199       36185
%        36198       36184
%        36197       36182
%        ....
% Number 36201 gets to 36197, which is not valid anymore after a few more
% removals
if not( isempty( vertChanges ))
    j1 = 1;
    while j1 < size( vertChanges, 1 )
        % The first change proposed is
        first_change = vertChanges( j1, 2 );
        no_changes = true;
        
        if first_change > 0        
            % Check if this change is further changed later
            for j2 = j1 + 1 : size( vertChanges, 1 )
                if vertChanges( j2, 1 ) == first_change
                    % Correct the first-change
                    vertChanges( j1, 2 ) = vertChanges( j2, 2 );
                    % Make current change void
                    vertChanges( j2, :) = -j1;
                    no_changes = false;
                    break;
                end
            end
        end
        
        if no_changes
            j1 = j1 + 1;
        end
    end

    % Remove void changes
    voidCH = find( vertChanges(:, 1) < 0 );
    if not( isempty( voidCH ))
        vertChanges( voidCH, : ) = [];
    end
end



