function subs = sample_global(X,nsamp)
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
d = ndims(X);
sz = double(size(X));
tsz = prod(sz); % Number of entries in X

%% Subscripts
subs = bsxfun(@(a,b)ceil(a.*b), rand(nsamp,d-1), sz(2:end));

%% Values
% vals = X(subs);

%% Weights
% wgts = tsz / nsamp * ones(nsamp,1);

