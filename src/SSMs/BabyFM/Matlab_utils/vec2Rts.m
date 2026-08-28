function [ R,t,s ] = vec2Rts(b,n)
%VEC2RST Convert vector representation of pose to R matrix, t vector, scale

r = b(1:3)';
t = b(4:4+n-1)';
s = b(4+n);
if (norm(r)==0) || ~isreal(r)
    R = eye(3);
else
    R = vrrotvec2mat([r./norm(r) norm(r)]);
end

end