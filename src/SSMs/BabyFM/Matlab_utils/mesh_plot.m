function h = mesh_plot( myMesh, varargin )

% h = plot_mesh( myMesh )
% - myMesh is a structure with fields 'verts' (3 * NV matrix) and 'faces'
% (3 * NF) triangles
% 
% h = plot_mesh( myMesh, scalarFunction )
% - scalarFunction is a vector of the same length as the number of
% vertices. It is a function over the mesh vertices and will be color-coded
% on top of the mesh.
% 
% h = plot_mesh( myMesh, vectorFunction )
% - Displays the vector function (must be also 3D) with arrows on top of
% the vertices
%

if size( myMesh.verts, 1 ) > 3
    error('Vertices must be organized as a 3 x NV matrix');
end
if size( myMesh.faces, 1 ) > 3
    error('Faces must be organized as a 3 x NF matrix');
end

fv = zeros( size( myMesh.verts, 2 ), 1);
fv3 = [];
scaleFactor = 0;
arrowDensity = 0;

if not( isempty( varargin ))
    arg1 = varargin{1};
    if length( arg1(:) ) == size( myMesh.verts, 2 ) ||...
            length( arg1(:) ) == size( myMesh.faces, 2 )
        fv = arg1(:);
    else
        if length( arg1(:) ) == length( myMesh.verts(:) )
            fv3 = arg1;
        end
    end
    while length( varargin ) > 1
        valid_args = 0;
        if strcmpi( varargin{2}, 'arrowSize' )
            scaleFactor = varargin{3};            
            valid_args = 2;
        end
        if strcmpi( varargin{2}, 'arrowDensity' )
            arrowDensity = varargin{3};
            valid_args = 2;
        end
        
        if not( valid_args )
            error('Unrecognized input arguments');
        else
            varargin( 2 : 1 + valid_args ) = [];
        end
    end
end

if length( fv ) == size( myMesh.faces, 2 )
    h = patch('vertices', myMesh.verts',...
        'faces', myMesh.faces', 'FaceVertexCData', fv, 'FaceColor', 'flat' );
else
    h = patch('vertices', myMesh.verts',...
        'faces', myMesh.faces', 'FaceVertexCData', fv, 'FaceColor', 'interp' );
end
lighting gouraud
 
%camlight infinite; 
camproj('perspective');
axis square; 
axis off;

%cameramenu

cameratoolbar;

axis tight;
axis equal;
if length( fv ) == size( myMesh.faces, 2 )
    shading flat;
else
    shading interp;
end
 camlight headlight;

        
if not( isempty( fv3 ))
    hold on;
    ss_idx = 1 : size( myMesh.verts, 2 );
    if arrowDensity 
        aDsq = arrowDensity * arrowDensity;
        for jv = 1 : length( ss_idx )
            if ss_idx( jv )
                dd2 = sum((myMesh.verts - ...
                    repmat( myMesh.verts(:,jv), [1 length( ss_idx )])).^2 );            
                ss_idx( dd2 < aDsq ) = 0;
                ss_idx( jv ) = jv;
            end
        end
        ss_idx( ss_idx == 0 ) = [];
    end
    h = quiver3(...
        myMesh.verts(1,ss_idx), myMesh.verts(2,ss_idx), myMesh.verts(3,ss_idx),...
        fv3(1,ss_idx), fv3(2,ss_idx), fv3(3,ss_idx),...
        scaleFactor, 'LineWidth', 2);

    if scaleFactor 
        try
            adjust_quiver_arrowhead_size( h, ( scaleFactor ).^(1/3));
        catch
            warning('Cannot set arrow scaleFactor');
        end
    end
    alpha .5; colormap gray;
end





