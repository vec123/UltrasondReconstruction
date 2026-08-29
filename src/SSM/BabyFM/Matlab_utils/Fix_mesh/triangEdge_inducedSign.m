function indSign = triangEdge_inducedSign( triangVerts, edgeVerts )

% indSign = triangEdge_inducedSign( triangVerts, edgeVerts )
%
% Computes the induced orientation from triangle defined at vertices
% triangVerts INTO the edge defined by edgeVerts
%
% faces defined as [v1 v2 v3] induce POSITIVE orientations on edges 
% [v1,v2], [v2,v3] and [v3,v1].
% This is because when we delete j-th vertex from a simplex, the 
% remaining face (in this case an edge) has a sign related to (-1)^j
% (recall j = 0,1...n-1), therefore v1-v3 has opposite sign to v1-v2-v3
%

if not(length( triangVerts ) == 3) || not(length( edgeVerts ) == 2)
    error('Unexpected length for input arguments');
end

i1 = find( triangVerts == edgeVerts(1) );
i2 = find( triangVerts == edgeVerts(2) );
if isempty(i1) || isempty(i2) || (i1 == i2)
    errro('Edge-triangle missmatch');
end

indSign = (2*(i1 < i2)-1) * (2*(abs(i1 - i2) == 1)-1);
% indSign = 1;
% if i1 < i2     % 1,2(+) or 1,3(-) or 2,3(+)
%     if i2 - i1 > 1       
%         indSign = -1;
%     end
% else           % 2,1(-) or 3,1(+) or 3,2(-)
%     if i1 - i2 == 1
%         indSign = -1;
%     end
% end

        
        
    