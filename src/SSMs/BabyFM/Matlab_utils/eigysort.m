function [V,D]=eigysort(X)
%[V,D]=eigysort(X)
%Calcula los autovecs/vals de X
%y los ordena en sentido decreciente
%de autovalores
[V,D]=eig(X);
[waste index]=sort(diag(D));
D=diag(sum(D(:,index(length(index):-1:1))));
V=V(:,index(length(index):-1:1));