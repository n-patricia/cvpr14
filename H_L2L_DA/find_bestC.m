function [ bestC ] = find_bestC( dataset )
%FIND_BESTC Summary of this function goes here
%   
% find best C 
%
% INPUT: dataset
% OUTPUT: bestC
load_settings;

addpath('classifier');

cvfile = ['data/',dataset,'/',dataset,'_bestC.mat'];


acc_array = zeros(numel(features),numel(C_array),n_fold,n_fold);

bestC = [];
fprintf('cross validation best-C...\n');
for f=1:numel(features)
load(['data/',dataset,'/',dataset,'.mat']);

for cc=1:numel(C_array)
    fprintf('.')
    for z=1:n_fold
    
    load(['data/',dataset,'/',dataset,'_split_F',num2str(z),'.mat']);

    Y = label(tr_idx);
    X = fts(tr_idx,:);
    
    hp.type = 'rbf';
    hp.gamma = 1/mean2(dist_euclid(X,X));
    model_init = k_init(@compute_kernel,hp,C_array(cc));
    model_init.n_cla = n_class;
    
    load(['data/',dataset,'/',dataset,'_cvsplit_F',num2str(z),'.mat']);    
    train_idx = split.train_idx;
    test_idx  = split.test_idx;

    for i=1:numel(train_idx);
        Ytrain = Y(train_idx{i});
        Ytest  = Y(test_idx{i});
        Xtrain = X(train_idx{i},:);
        Xtest  = X(test_idx{i},:);
        
        
        [model,loo_err,loo_pred] = kls_train_multi(Xtrain',Ytrain',model_init);
        [pred,vals] = k_predict_multi(Xtest',model);
        [mx,y] = max(vals',[],2);
        
        acc_array(f,cc,z,i) = numel(find(Ytest==y))/numel(y);
    end
    end


end

clear fts label
fprintf('\n');
end
% find best C
mat = mean(mean(acc_array,4),3);
for f=1:numel(features)
[a,b] = max(mat(f,:));
bestC(f) = C_array(b);
end
save(cvfile,'bestC','acc_array');
end

