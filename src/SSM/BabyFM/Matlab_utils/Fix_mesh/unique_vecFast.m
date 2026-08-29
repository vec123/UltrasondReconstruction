function b = unique_vecFast( a )
%
% Fast version of unique() that only accepts colum-vectors
%

if not( size( a, 2 ) == 1 )
    error('Only single-colum vectors accepted');
end
  
numelA = numel(a);
b = sort( a );
db = diff(b);

% if (isnan(db(1)) || isnan(db(numelA-1)))
%     d = b(1:numelA-1) ~= b(2:numelA);
% else
    d = db ~= 0;
% end
 
d(numelA,1) = true;
b = b(d);

