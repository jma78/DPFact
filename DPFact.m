%% Jing Ma
% DPFact
clc; close all; clearvars;
addpath(genpath('./tensor_toolbox'));
[X, Xs, dim, nonzero_ratio, K, cutoffs, Pcutoffs, Dcutoffs] = genTensorByFiles; % generate tensor from the input files

%% Server: set algorithm parameters
rank = 50;
maxepoch = 50;
epciters = 1;
tau = 2; % local update times
rho = 5; % quadratic penalty term
eta_p = 0.001;  % learning rate for local SGD
rmses=zeros(1, maxepoch);  %the rmse of each epoch
communication = 0;
lambda_list = [1,1.8,3.2,1.8,1.5,0.6];
eta_p1 = eta_p;
eta_p2 = eta_p/10^8;
eta_p3 = eta_p2;
nd = length(dim);
nsamplsq = 10;
gsamp = ceil(nsamplsq * (sum(size(X))/nd));

%% Set clients workspace
for k=1:K
    client(k).X=Xs{k}; % observed tensor
end

%% Non zero elements index of Observed Tensor (X)
Kindices = cell(1,K);
for k=1:K
    Kindices{k} = [client(k).X.subs, client(k).X.vals];
end

GmatrixInit = cell(1,3); % Initial Global factor matrix
Gmatrix = cell(1,3);

%% Server: initialization (mode 2 and 3)
for n = [2,3,1] % n=1 initialize at institutions
    GmatrixInit{n} = rand(dim(n),rank);
end

rho2 = rho;
rho3 = rho;

%%
add_dps = {'on'};
add_l21norms = {'on'};
for i1 = 1:length(add_dps)
    for i2 = 1:length(add_l21norms)
        add_dp = add_dps{i1};
        l21norm = add_l21norms{i2};
        Gmatrix{1} = GmatrixInit{1};
        Gmatrix{2} = zeros(dim(2),rank);
        Gmatrix{2}(Pcutoffs{1},:)=GmatrixInit{2}(Pcutoffs{1},:);
        Gmatrix{3} = zeros(dim(3),rank);
        Gmatrix{3}(Dcutoffs{1},:)=GmatrixInit{3}(Dcutoffs{1},:);

        %% Hospitals: compute statistics for X / initialization
        for k=1:K
            client(k).dim=size(client(k).X); %data
            client(k).Ai=cell(1,3); % initialize 3 factor matrices
            % n = 1
            client(k).Ai{1}=zeros(dim(1),rank);
            client(k).Ai{1}(cutoffs{k},:) = Gmatrix{1}(cutoffs{k},:);

            client(k).Ai{2}=Gmatrix{2};
            client(k).Ai{3}=Gmatrix{3};

        end

        T = ktensor(Gmatrix);
        normresidual= double(norm(plusKtensor(X, -T)));
        old_rmse=double(normresidual/(sqrt(nnz(X))));
        disp([epoch, 0, old_rmse])

        %% main loop
        for epoch = 1:maxepoch
            % do local update;
            Ai = cell(1,K);
            tic;
            subs2 = sample_global(X, gsamp);
            parfor k=1:K
                subs1 = sample_local(cutoffs{k},gsamp);
                gsubs = [subs1,subs2];
                xnzidx = tt_sub2ind(size(client(k).X),gsubs);
                gvals = client(k).X(xnzidx);
                lambda = lambda_list(k);
                client(k).Ai= LocalUpdate(gsubs,gvals, epciters,tau, Gmatrix, client(k).Ai, rho2, rho3, eta_p1, eta_p2, eta_p3, cutoffs{k}, Pcutoffs{k}, Dcutoffs{k}, client(k).X, rank, add_dp, lambda, l21norm);
                Ai{k} = client(k).Ai;
            end
            
            tmp_G1 = zeros(dim(1), rank);
            for k=1:K
                tmp_G1=tmp_G1+client(k).Ai{1};
            end
            Gmatrix{1}= tmp_G1;
            % update global factor matrix
            for n=2:3
                tmp_Gn = zeros(dim(n), rank);
                if n==2
                    for k=1:K
                        diff = rho2 * (client(k).Ai{n}-Gmatrix{n});
                        tmp_Gn=tmp_Gn+diff;
                    end
                    gradient = tmp_Gn;
                    Gmatrix{n} = Gmatrix{n} + eta_p2 * gradient;
                else
                    for k=1:K
                        diff = rho3 * (client(k).Ai{n}-Gmatrix{n});
                        tmp_Gn=tmp_Gn+diff;
                    end
                    gradient = tmp_Gn;
                    Gmatrix{n} = Gmatrix{n} + eta_p3 * gradient;
                end
            end

            % compute the result every epoch
            T = ktensor(Gmatrix);
            normresidual= norm(plusKtensor(X, -T));
            rmses(epoch)= normresidual/(sqrt(nnz(X)));
            disp([epoch, toc, rmses(epoch)])
            old_rmse = rmses(epoch);

        end
    end
end