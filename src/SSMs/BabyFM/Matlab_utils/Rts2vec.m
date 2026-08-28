function b = Rts2vec( R,t,s )
%RST2VEC Convert R matrix, t vector, scale to vector representation of pose

r = vrrotmat2vec(R);
b(1:length(r)-1)=(r(1:3)./norm(r(1:3))).*r(4);
b(length(r) : length(r) + length(t) - 1)=t;
b(length(r) + length(t) )=s;

end