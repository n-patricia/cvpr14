function create_targetout( dataset )
%CREATE_TARGETOUT Summary of this function goes here
% create target output confidence for binary transfer learning
% INPUT: dataset
% OUTPUT: target confidence file

% This code is part of the CVPR 2014 paper
% "Learning to Learn, from Transfer Learning to Domain Adaptation: A Unifying Perspective", 
% N.Patricia, B. Caputo (* equal authorship). 

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

fprintf('loading ''%s''\n',dataset);
load(dataset);
n = numel(names);
savefile = strrep(dataset,'.mat','_target_out.mat');

n_fold = 5;
for i=1:n
    [split1, split2] = create_split(t{i,1},n_fold);
    tr_split{i,1} = split1;
    te_split{i,1} = split2;
end

load_settings;

fprintf('computing target output confidence...\n');
for i=1:n
    fprintf('SUBJ:%d\n',i);
    
    hp.type = 'rbf';
    hp.gamma = 1/mean2(dist_euclid(p{i},p{i}));    
    
    for cc=1:numel(C_array)
    for f=1:numel(feat)
    for k=1:n_fold
        idx_tr = tr_split{i,1}{k};
        p_tr = p{i,f}(idx_tr,:);
        t_tr = t{i,f}(idx_tr);
        nY = length(t_tr');
        
        model_init =  k_init(@compute_kernel, hp, C_array(cc));

        Ktrain = feval(model_init.ker, p_tr', 1:nY, 1:nY, model_init.kerparam);
        model_init.X = p_tr;
        model_init.K = Ktrain;
        model_init.Y = t_tr';
        model_init.S = [1:nY];
        [model_loo{k,f,cc}, loo_err, loo_pred] = kls_train_K(model_init,0);
        clear Ktrain
        clear model_init
    end
    end
    end
    
    clear hp
    clear err_tmp Ktest cue out_conf
    
    for f=1:numel(feat)
    for cc=1:numel(C_array)
        out_conf{cc} = zeros(numel(t{i,1}),1);
        for k=1:n_fold
            clear pred
            idx_te = te_split{i,1}{k};
            cue.p_test = p{i,f}(idx_te,:);
            cue.t_test = t{i,f}(idx_te);

            Ktest = utility_predict_single(cue, model_loo{k,f,cc});
            [pred,vals] = k_predict_K(Ktest, model_loo{k,f,cc});
            ypred(idx_te) = pred;
            out_conf{cc}(idx_te) = vals';
        end

        err_tmp(cc) = numel(find(ypred==t{i}'))/numel(t{i});
        clear ypred
    end
        
    [V,I] = max(err_tmp);
    t_out_conf{i,1}(:,f) = out_conf{I};
    
    end
    
    clear model_loo
    
end

fprintf('done, saving ''%s''\n\n',savefile);
save(savefile,'t_out_conf','p','t','feat','names');

end

