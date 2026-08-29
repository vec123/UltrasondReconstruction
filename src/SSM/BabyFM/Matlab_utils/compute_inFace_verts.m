function inFace = compute_inFace_verts(FaceModel)



meanMesh.verts = reshape(FaceModel.meanShape, [3,length(FaceModel.meanShape)/3]);
meanMesh.faces = FaceModel.triang;

% eyes_lmks = FaceModel.landmark_verts( contains( FaceModel.landmark_names, {'exR','enR','n','enL','exL'} ) );
oiR_ind = FaceModel.landmark_verts( contains( FaceModel.landmark_names, 'oiR' ) );
lmks_ind = FaceModel.landmark_verts;


% lower_eyes = find( meanMesh.verts(2,:) < ( max(meanMesh.verts(2,eyes_lmks))...
%                      + max(meanMesh.verts(2,:)) ) / 2 );

ears = find( meanMesh.verts(3,:) < meanMesh.verts(3,oiR_ind) + ...
       (meanMesh.verts(3,lmks_ind(8)) - meanMesh.verts(3,oiR_ind))/10 );

% inFace = setdiff(lower_eyes, ears);
inFace = setdiff(1:size(meanMesh.verts,2), ears);


% FOREHEAD and NECK
upper_pt = meanMesh.verts(:,lmks_ind(3)) + [0;2.5*1e-3;0]; % 3*1e-3;0];
lower_pt = meanMesh.verts(:,lmks_ind(19))- [0;2*1e-3;0];


% NECK
oiR = meanMesh.verts(:,oiR_ind);
slope = (oiR(3)-lower_pt(3))/(oiR(2)-lower_pt(2));
intercept = lower_pt(3)-lower_pt(2)*slope;
neck = find( meanMesh.verts(3,:) < slope*meanMesh.verts(2,:)+intercept );
inFace = setdiff(inFace,neck);


% FOREHEAD
tR = meanMesh.verts(:,lmks_ind(20));
slope = (tR(3)-upper_pt(3))/(tR(2)-upper_pt(2));
intercept = upper_pt(3)-upper_pt(2)*slope;
forehead = find( meanMesh.verts(3,:) < slope*meanMesh.verts(2,:)+intercept );


% eval(['inFace_',model_name,' = setdiff(inFace, forehead);']);
inFace = setdiff(inFace, forehead);
end