function h = plot3pts( XYZ, varargin )

% plot3point( XYZ )
%
% This function is just a wrapper to plot3 so that we pass a single
% parameter: the matrix XYZ, of size 3 * N (therefore, each COLUMN
% is a point in 3D
%

if not( size( XYZ, 1 ) == 3 )
    error('Input matrix muxt be of size 3*N');
end

h = plot3( XYZ(1,:), XYZ(2,:), XYZ(3,:), varargin{:});
