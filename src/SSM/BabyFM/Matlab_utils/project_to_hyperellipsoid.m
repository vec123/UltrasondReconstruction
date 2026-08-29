function [b_HE] = project_to_hyperellipsoid(b,eigVals,beta,options)
% Shape Constraint Strategies: Novel Approaches and Comparative Robustness
% Juan J. Cerrolaza, Arantxa Villanueva, Rafael Cabeza

% SOLVE
% sum(b_new^2/(beta^2*eigVals)) - 1 = 0
% min( sum( (b_new - b).^2 ) )

tol = 1e-8; 
alpha1 = 0;
max_iter = 1000;
verbose = 0;

if nargin > 3
    if isfield(options,'tol'), tol = options.tol; end

    if isfield(options,'alpha0'), alpha1 = options.alpha0; end

    if isfield(options,'max_iter'), max_iter = options.max_iter; end
    
    if isfield(options,'verbose'), verbose = options.verbose; end
end


f = @(alpha) sum( (b.^2 .* beta^2 .* eigVals) ./ (eigVals.*beta^2 + alpha*ones(size(eigVals))).^2  ) - 1 ;

df = @(alpha) - sum( (2*beta^2.*(b.^2).*eigVals) ./ (eigVals.*beta^2 + alpha*ones(size(eigVals))).^3 );

alpha0 = Inf;
alphas = [];
it = 0;
aug_alpha = Inf;
while aug_alpha > tol && it < max_iter
    it = it + 1;
    alpha0 = alpha1;
    if abs(df(alpha0)) < 1e-15, break; end
    alpha1 = alpha0 - f(alpha0)/df(alpha0);
    
    if abs(alpha0 - alpha1) == aug_alpha, break; end
    % this happens when the values of alpha0 and alpha1 are being
    % alternated constantly
    
    aug_alpha = abs(alpha0 - alpha1);
    
    if verbose
        fprintf('it %i - alpha_k = %g - alpha_(k+1) = %g - aug_alpha = %g\n', it, alpha0, alpha1, aug_alpha)
    end
    
    alphas = [alphas, alpha0];
end

b_HE = (beta^2.*b.*eigVals) ./ (eigVals.*beta^2 + alpha1.*ones(size(eigVals)));

end

