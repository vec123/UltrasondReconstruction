function [d12_v, d12_idx, d21_v, d21_idx] = pointCloud_p2p_distance( pc1, pc2, varargin )
% Inputs are of size 3 x Num_points

k = 1;
if ~isempty(varargin)
    k = varargin{1};    
end

myKdTree = createns( pc2','nsmethod','kdtree');
[d12_idx, d12_v] = knnsearch(myKdTree, pc1','k',k);

myKdTree1 = createns( pc1','nsmethod','kdtree');
[d21_idx, d21_v] = knnsearch(myKdTree1, pc2','k',k);