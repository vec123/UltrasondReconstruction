function [U,D,media]=pca_ld(Data,varargin)

%Calcula PCA en los datos de Data, organizados
%por columnas. U: auvectores. D: autovalores.
%ld significa low-dimensional: los datos son
%de baja dimension (no como en el caso de las
%imagentes), luego dim(X*X')<dim(X'*X) y es
%algo más fácil...

%Numero de vectores a los que se aplica PCA.
M=size(Data,2);
%Centramos en la media.
media=(mean(Data'))';
X=Data-(media*ones(1,M));
%Matriz de covarianzas: (1/M)*X*X'.
%Cova=(1/M)*X.*X';

[U,D]=eigysort((1/M)*(X*X.'));
%CovMAT = (1/M)*(X*X.');
%CovMAT = 
%[U,D]=eigysort(CovMAT);

