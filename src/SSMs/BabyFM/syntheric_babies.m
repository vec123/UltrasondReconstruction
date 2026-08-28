rsp = @(x) reshape(x, 3,[]);
% LOAD BABY MODEL
addpath(genpath('Matlab_utils'))
load('BabyFaceModel.mat');
FaceModel=BabyFaceModel;

% Baby morphable model characteristics

options.var=98;
var= '_var_98';
options.chi_squared=0.9999999999999999;
options.ndims = find(cumsum(FaceModel.pctVar_per_eigen) > options.var,1 );
options.shapeMU = reshape(FaceModel.refShape,1,[]);
options.lmks_vertsIND = FaceModel.landmark_verts;
options.shapeEV= FaceModel.eigenValues(1:options.ndims);
options.shapePC= FaceModel.eigenFunctions(:,1:options.ndims);
options.trilist= FaceModel.triang;

% New model
options.meanDeformation= FaceModel.meanDeformation;
options.scale_factor = sqrt(size(FaceModel.refShape,2));

% mesh structere creation mesh.verts and mesh.faces
 mean_mesh.verts= reshape(options.shapeMU,3,[]); % reshape the mean to put in 3xN vertex representation
 mean_mesh.faces= double(options.trilist); % mesh tringulation (+ 1 python elements that are 0 )

% MEAN + LANDMARKS MODEL 
figure;
mesh_plot(mean_mesh);
material([0.3 0.7 0]);
colormap([0.9 0.9 0.9]);
hold on;
plot3(mean_mesh.verts(1,options.lmks_vertsIND(:)),mean_mesh.verts(2,options.lmks_vertsIND(:)),mean_mesh.verts(3,options.lmks_vertsIND),'*r')
text(mean_mesh.verts(1,options.lmks_vertsIND(:)),mean_mesh.verts(2,options.lmks_vertsIND(:)),mean_mesh.verts(3,options.lmks_vertsIND) + 0.001,BabyFaceModel.landmark_names,'FontSize',14)

%% GENERATE SYNTHETIC DATASET
nOfSamples = 15;
chi_squared = 0.99;
var = 99;
nOfModes = find( cumsum(FaceModel.pctVar_per_eigen) > var,1 );

beta2 = chi2inv(chi_squared,nOfModes);

b = FaceModel.eigenValues(1:nOfModes).* (-3 + (3+3)*rand(nOfModes,nOfSamples))*10^6;
%b = b';
dMah2 = diag(b' * diag( 1./FaceModel.eigenValues(1:nOfModes) ) * b);

% % Remove faces outside hyperellipsoid
% ind_b_outside = dMah2 > beta2;
% while any(ind_b_outside)
%     fprintf('%i\n', sum(ind_b_outside))
%     new_b = mvnrnd(zeros(nOfModes,1),diag(FaceModel.eigenValues(1:nOfModes)),sum(ind_b_outside));
%     b(:,ind_b_outside) = new_b';
%     
%     dMah2 = diag(b' * diag( 1./FaceModel.eigenValues(1:nOfModes) ) * b);
%     ind_b_outside = dMah2 > beta2;
% end

for i= 1:15%nOfSamples
    aux = FaceModel.meanDeformation'+ options.shapeMU'+ (FaceModel.eigenFunctions(:,1:nOfModes)*b(:,i));
    rec =  rsp(aux);

    beta2 = chi2inv(chi_squared,nOfModes);
    figure; 
    mesh_s.verts=rec;
    mesh_s.faces = double(options.trilist);
    mesh_plot(mesh_s)
    material([0.3 0.7 0]);
    colormap([0.9 0.9 0.9]);

end