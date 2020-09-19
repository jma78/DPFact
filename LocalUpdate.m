%% Jing Ma
% Permutation-based SGD for mimic-iii data
function Lmatrix= LocalUpdate(gsubs,gvals, epciters,tau, Gmatrix, Lmatrix, rho2, rho3, eta_p1, eta_p2, eta_p3, cutoffs, Pdim, Ddim, Ltensor, rank, add_dp, lambda, l21norm)
%% main loop
for i=1:tau
    for iter = 1: epciters
        for p = 1:length(gsubs)
            u = gsubs(p,1);
            v = gsubs(p,2);
            w = gsubs(p,3);
            R = double(gvals(p));
            gradient = (Lmatrix{1}(u,:)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:))'-R)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:));
            Lmatrix{1}(u,:) = Lmatrix{1}(u,:) - eta_p1*gradient;
            gradient = (Lmatrix{1}(u,:)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:))'-R)*(Lmatrix{1}(u,:).*Lmatrix{3}(w,:)) + (rho2) * (Lmatrix{2}(v,:)-Gmatrix{2}(v,:));
            Lmatrix{2}(v,:) = Lmatrix{2}(v,:) - eta_p2*gradient;
            gradient = (Lmatrix{1}(u,:)*(Lmatrix{2}(v,:).*Lmatrix{3}(w,:))'-R)*(Lmatrix{1}(u,:).*Lmatrix{2}(v,:)) + (rho3) * (Lmatrix{3}(w,:)-Gmatrix{3}(w,:));
            Lmatrix{3}(w,:) = Lmatrix{3}(w,:) - eta_p3*gradient;
        end
        for r =1:rank
            Lmatrix{1}(:,r) = Lmatrix{1}(:,r) * max(0,(1-lambda/(norm(Lmatrix{1}(:,r)))));
        end
    end
end

if strcmp(add_dp, 'on')
    %% Calculate L2 and L3 based on factor matrix
    gradient = cell(1,2);
    for n=2:3
        piitpii=ones(rank,rank);
        for nn=[1:n-1, n+1:3]
            piitpii=piitpii .*(Lmatrix{nn}' * Lmatrix{nn});
        end
        term1 = mttkrp(Ltensor, Lmatrix,n);
        term2 = Lmatrix{n} * piitpii;
        gradient{n-1} = -term1+term2;
    end
    
    l2 = norm(gradient{1});
    l3 = norm(gradient{2});
    
    %% Add Gaussian noise to achieve (epsilon, delta)-differential privacy
    % calculate l2_sensitivity for each factor matrix
    l2_sensitivity_L2 = 2*tau*l2*eta_p2;  % l2 sensitivity for mode 2 factor matrix
    l2_sensitivity_L3 = 2*tau*l3*eta_p3;  % l2 sensitivity for mode 3 factor matrix
    
    % Assign Privacy Budget for this hospital
    epsilon_P = 0.35;
    epsilon_D = 0.35;
    delta_P = 0.00001;
    delta_D = 0.00001;
    
    % add noise to each row of mode2 matrix(P)
    c_P = sqrt(2*log(1.25/delta_P));
    sigma_P = c_P*l2_sensitivity_L2/epsilon_P;   % calibrate noise
    var_P = sigma_P^2;
    noise_matrix2 = var_P*randn(size(Lmatrix{2}));
    Lmatrix{2} = Lmatrix{2} + noise_matrix2;
        
    % add noise to each row of mode3 matrix(D)
    c_D = sqrt(2*log(1.25/delta_D));
    sigma_D = c_D*l2_sensitivity_L3/epsilon_D;  % calibrate noise
    var_D = sigma_D^2;
    noise_matrix3 = var_D*randn(size(Lmatrix{3}));
    Lmatrix{3} = Lmatrix{3} + noise_matrix3;
        
end

end

function subs = sample_global(X,nsamp)
d = ndims(X);
sz = double(size(X));
subs = bsxfun(@(a,b)ceil(a.*b), rand(nsamp,d-1), sz(2:end));
end

function subs = sample_local(cutoff,nsamp)
sz = size(cutoff);
idx = bsxfun(@(a,b)ceil(a.*b), rand(nsamp,1), sz(1));
subs = cutoff(idx);
end