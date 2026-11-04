function setupGeom3d(varargin)
%SETUPGEOM3D  One-line description here, please.
%
%   output = setupGeom3d(input)
%
%   Example
%   setupGeom3d
%
%   See also
%
 
% ------
% Author: David Legland
% e-mail: david.legland@inra.fr
% Created: 2018-11-09,    using Matlab 9.5.0.944444 (R2018b)
% Copyright 2018 INRA - Cepia Software Platform.

% extract library path
fileName = mfilename('fullpath');
libDir = fileparts(fileName);

moduleNames = {...
    'geom3d', ...
    'meshes3d'};

disp('Installing geom3d Library');
addpath(libDir);

% add all library modules
for i = 1:length(moduleNames)
    name = moduleNames{i};
    fprintf('Adding module: %-20s', name);
    addpath(fullfile(libDir, name));
    disp(' (ok)');
end
