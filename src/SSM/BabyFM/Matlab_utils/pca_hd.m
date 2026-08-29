function [U,D,media]=pca_hd(Data)

%Calcula PCA en los datos de Data, organizados
%por columnas. E: auvectores. D: autovalores.

%Numero de vectores a los que se aplica PCA.
M=size(Data,2);
%Centramos en la media.
media=(mean(Data'))';
X=Data-(media*ones(1,M));
%Matriz de covarianzas: (1/M)*X*X'.
%Empezamos buscando autovecs/vals de X'*X
%[V,D]=eig(X.'*X);
[V,D]=eigysort(X.'*X);
%Los autovectores estan normalizados, pero
%los autovalores no estan ordenados. Los ordenamos
%y eliminamos autovalor/vector menor por ser nulo...
V=V(:,1:end-1);
D=D(1:end-1,1:end-1);

%Autovectores de X*X'. X*V da autovectores de X*X', pero
%no de norma unidad... raiz(D) se usa para normalizar. Los
%autovectores de X*X' son los mismo que los de (1/M)*X*X'
%(multiplicar por constante afecta a autovalores, pero no
%a autovectores de modulo unidad...).
U=X*(V.*repmat(sqrt(1./diag(D))',[M 1]));
%D da autovectores de X*X' y X'*X (coinciden). Pero el
%factor 1/M sí afecta...
D=(1/M)*D;