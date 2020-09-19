function subs = sample_local(cutoff,nsamp)
%TT_SAMPLE_UNIFORM Uniformly sample indices from a tensor.
%
%   [SUBS,VALS,WGTS] = TT_SAMPLE_UNIFORM(X,N) samples N indices uniformly at
%   random from X, along with the corresponding values and the weight of
%   the sample. This is for use with stochastic optimization in GCP_OPT.
%
%   See also GCP_OPT, TT_SAMPLE_STRATIFIED.
%
%MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.

% Created by Tamara G. Kolda, Fall 2018. Includes work with
% collaborators David Hong and Jed Duersch. 


%% Setup
sz = size(cutoff);

%% Subscripts
idx = bsxfun(@(a,b)ceil(a.*b), rand(nsamp,1), sz(1));
subs = cutoff(idx);

%% Values
% vals = X(subs);

%% Weights
% wgts = tsz / nsamp * ones(nsamp,1);

