function hl2l_da( sources, target )
%HL2L_DA Summary of this function goes here
%   Detailed explanation goes here
% High Level-Learning2Learn for domain adaptation problem,
% with prior-features as baseline
% 
% INPUT: sources, target
% OUTPUT: classification accuracy
%
% This code is part of the CVPR 2014 paper
% "Learning to Learn, from Transfer Learning to Domain Adaptation: A Unifying Perspective", 
% N.Patricia, B. Caputo (* equal authorship). 
%
% Copyright (C) 2013-2014, Novi Patricia
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% Contact the authors: novi.patricia [at] idiap.ch


load_settings;
addpath('../classifier');

priorfile = ['prior/',target,'/',target];
% load lssvm classifier
for s=1:numel(sources)
    priorfile = [priorfile,'_',sources{s}];
end
priorfile = [priorfile,'_prior.mat'];
if ~exist(priorfile,'file')
    fprintf('this prior is not existed!\n');
    fprintf('compute ''%s'' first\n',priorfile);
    return
end
prior = load(priorfile);

resdir = ['result/',target,'/'];
if ~exist(resdir,'dir'), system(['mkdir -p ',resdir]); end
done_file = [resdir,target];

for s=1:numel(sources)
    done_file = [done_file,'_',sources{s}];
end
done_file = [done_file,'_avg.mat'];

fprintf('computing high level cue integration...\n');
for i=1:n_fold
    fprintf('fold:%d\n',i);
    
    Ctrain = [];
    Ctest  = [];
    P = [];
    
    
    load(['data/',target,'/',target,'_split_F',num2str(i),'.mat']);        
    tr = [];
    te = [];
    for k=1:n_class
        idx = find(te_label==k);
        tr = [tr; idx(1:min(n_train,numel(idx)-1))];
        te = [te; idx(min(n_train,numel(idx)-1)+1:end)];
    end
    Ytrain = te_label(tr);
    Ytest  = te_label(te);
    
    assert(numel(unique(Ytrain))==n_class);
    assert(numel(unique(Ytest))==n_class);
    
    for f=1:numel(features)
        % prior confidence of source
        for s=1:numel(sources)
            P = [P prior.s_out_conf{s,f,i}]; 
        end
        
        % cue integration        
        Ctrain = [Ctrain prior.t_out_conf{f,i}(tr,:)];
        Ctest  = [Ctest prior.t_out_conf{f,i}(te,:)];
        
    end
    
    % prior prediction
    Ptrain = P(tr,:);
    Ptest  = P(te,:);
    
    % concantenate prior and target prediction
    Ctrain = [Ctrain Ptrain];
    Ctest  = [Ctest Ptest];
    
    % Prior-Features
    clear err_tmp
    for cc=1:numel(C_array)
        model_pr = linear_train(Ytrain, sparse(Ptrain), ['-B 1 -s 4 -c ' num2str(C_array(cc))]); 
        [yhat, acc, dec] = linear_predict(Ytest, sparse(Ptest), model_pr);
        err_tmp(cc) = acc(1);
        clear model_pr
    end
    [V,I] = max(err_tmp);
    pr_acc(i) = err_tmp(I);
    
        
    % H-L2L (SVM-DAS)
    clear err_tmp
    hp.type = 'rbf';
    hp.gamma = 1/mean2(dist_euclid(Ctrain,Ctrain));
    K = compute_kernel(Ctrain',Ctrain',hp);
    
    hp.gamma = 1/mean2(dist_euclid(Ctrain,Ctest));
    Ktest = compute_kernel(Ctrain',Ctest',hp);
    
    for cc=1:numel(C_array)
        model_da = svm_train_ova(Ytrain, [ (1:numel(Ytrain))' K ], ['-t 4 -c ' num2str(C_array(cc))]);
        [pred, acc, vals] = svm_predict_ova(Ytest, [(1:numel(Ytest))', Ktest'], model_da);    
        err_tmp(cc) = acc(1);
        clear model_da;
    end
    [V,I] = max(err_tmp);
    da_acc(i) = err_tmp(I);
end

fprintf('done, saving file ''%s''\n',done_file);
save(done_file,'*_acc');
end

