function create_prior( sources, target )
%CREATE_PRIOR Summary of this function goes here
%   Detailed explanation goes here
% create prior output confidence of sources and target
%
% INPUT: sources, target
% OUTPUT: prior file

load_settings;
addpath('../classifier');

priordir = ['prior/',target,'/'];
if ~exist(priordir,'dir'), system(['mkdir -p ',priordir]); end
priorfile = [priordir,target];
for s=1:numel(sources)
    priorfile = [priorfile,'_',sources{s}];
end
priorfile = [priorfile,'_prior.mat'];

% create source model
fprintf('creating source model...\n');
for s=1:numel(sources)
    
    for f=1:numel(features)
    load(['data/',sources{s},'/',sources{s},'_bestC.mat']);    
    % best C
    c = bestC;
        
    % load source
    load(['data/',sources{s},'/',sources{s},'.mat']);
    
    for i=1:n_fold
        % load source        
        load(['data/',sources{s},'/',sources{s},'_split_F',num2str(i),'.mat']);
        
        % source dataset
        % train feat
        t_train = tr_label;        
        p_train = fts(tr_idx,:);
        
        source_fts{s,f,i} = fts(tr_idx,:);
                
        % model ls-svm
        hp.type = 'rbf';
        hp.gamma = 1/mean2(dist_euclid(p_train,p_train));
        
        model = k_init(@compute_kernel, hp, c);
        model.n_cla = n_class;
        [model_lssvm_source{s,f,i}, loo_err, loo_pred] = kls_train_multi(p_train', t_train', model);
        
        clear model p_train t_train
        clear tr_* te_*
    end
    
    end
    
end
clear bestC
clear fts label

best_c = [];
load(['data/',target,'/',target,'_bestC.mat']);

% create target model
fprintf('creating target model...\n');
for f=1:numel(features)
% best C
best_c(f) = bestC;

load(['data/',target,'/',target,'.mat']);

for i=1:n_fold
    % load target
    load(['data/',target,'/',target,'_split_F',num2str(i),'.mat']);
    
    test_label{i} = te_label;
    test_idx{i} = te_idx;    
    test_fts{f,i} = fts(te_idx,:);
    
    % target train
    p_train = fts(tr_idx,:);
    t_train = tr_label;
    
    % create model ls-svm
    hp.type = 'rbf';
    hp.gamma = 1/mean2(dist_euclid(p_train,p_train));
    model = k_init(@compute_kernel, hp, best_c(f));
    model.n_cla = n_class;
    [model_lssvm_target{f,i}, loo_err, loo_pred] = kls_train_multi(p_train', t_train', model);    
    
    clear model p_train t_train
    clear tr_* te_*
end
clear fts    

end


% compute output confidence
fprintf('computing output confidence...\n');
for i=1:n_fold
   
    for f=1:numel(features)
        p_test = test_fts{f,i};
        t_test = test_label{i};

        for s=1:numel(sources)
            %s_out_conf{s,f,i} = zeros(size(p_test,1),n_class);
            tmp_conf = zeros(size(p_test,1),n_class);
            for j=1:n_fold                
                [pred, vals] = k_predict_multi(p_test',model_lssvm_source{s,f,j});
                tmp_conf = tmp_conf+vals';
            end
            s_out_conf{s,f,i} = tmp_conf./n_fold;
        end

        t_out_conf{f,i} = zeros(size(p_test,1),n_class);
        [pred, vals] = k_predict_multi(p_test', model_lssvm_target{f,i});
        t_out_conf{f,i} = vals';
    end
    
    
end

fprintf('done, save ''%s''\n',priorfile);
save(priorfile,'s_out_*','t_out_*','best_c');

end