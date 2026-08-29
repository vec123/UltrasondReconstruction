function L = cellArray_getSubLenghts( in_cellArray );

% This function is meant to compute the lenghts of sub-cell arrays (1-level
% only) of a given cell array
% Example
% Let u be a cell array as follow:
%     [4x1 double]
%     [3x1 double]
%     [4x1 double]
%     [4x1 double]
%     [7x1 double]
% Then, length( u ) = 6
% But, cellArray_getSubLenghts( u ) = [4, 3, 4, 4, 7]
%

L = zeros( 1, length( in_cellArray ));
for k = 1 : length( L )
    L(k) = length( in_cellArray{k} );
end
