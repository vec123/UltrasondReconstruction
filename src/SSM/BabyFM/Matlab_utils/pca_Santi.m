function [U,D,media]=pca_Santi(Data)

%[U,D,media]=pca(Data)

%Calcula PCA en los datos de Data, organizados
%por columnas. E: auvectores. D: autovalores.

[N,M]=size(Data);
if (N>=M-1),
   [U,D,media]=pca_hd(Data);
else
   [U,D,media]=pca_ld(Data);
end

