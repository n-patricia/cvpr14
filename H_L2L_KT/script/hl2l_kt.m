function hl2l_kt( dataset )
%HL2L_KT Summary of this function goes here
% High Level-Learning2Learn for binary transfer learning 
% with no-transfer and prior-features as baseline
% INPUT: dataset
% OUTPUT: accuracy result

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

load_settings;

TO = load(strrep(dataset,'.mat','_target_out.mat'));
SO = load(strrep(dataset,'.mat','_source_out.mat'));
n = numel(TO.names);
feat = TO.feat;

% concantenate target and source output confidence
for i=1:n
    out_conf{i} = TO.t_out_conf{i};
    
    prior_conf{i} = [];
    for ii=1:n
        if ii==i, continue; end
        prior_conf{i} = [prior_conf{i} SO.s_out_conf{i,ii}];        
    end

    out_conf{i} = [out_conf{i} prior_conf{i}];
end

load(dataset);
savefile = strrep(dataset,'.mat','_done.mat');

iter = 10;
step = 1;
n_step = 10;

fprintf('running H-L2L...\n');
for i=1:n
    fprintf('SUBJ:%d\n',i);
    
    for j=1:iter
        clear yrp orp prp
        
        % random permutation
        rp = randperm(numel(TO.t{i}));
        yrp = TO.t{i,1}(rp);
        orp = out_conf{i}(rp,:);
        prp = prior_conf{i}(rp,:);
        
        for f=1:numel(feat)
           xrp{f} = TO.p{i,f}(rp,:); 
        end
        
        pos = find(yrp==1);
        neg = find(yrp==-1);
        
        for s=1:n_step
            idx_tr = [neg(1:step*n_step); pos(1:step*s)];
            idx_te = [neg(31:end); pos(31:end)];
            
            Y = yrp(idx_tr);
            Ytest = yrp(idx_te);            
            
            % No-Transfer
            clear p_train_small t_train_small cue
            for f=1:numel(feat)
                p_train_small{f} = xrp{f}(idx_tr,:);
                t_train_small{f} = yrp(idx_tr);
                
                cue{f}.p_test = xrp{f}(idx_te,:);
                cue{f}.t_test = yrp(idx_te);
            end
            
            clear Ktr model         
            nY = length(t_train_small{1}');
            % computing model for No-Transfer
            hp.type = 'rbf';
            hp.gamma = 100;
            model = k_init(@compute_kernel,hp,1);                
            for f=1:numel(feat)
                Ktr(:,:,f) = feval(model.ker, p_train_small{f}', 1:nY, 1:nY, model.kerparam);
                model.XX{f} = p_train_small{f};
            end

            model.K = mean(Ktr,3);
            model.Y = t_train_small{1}';
            model.S = [1:nY];
            
            clear err_tmp
            for cc=1:numel(C_array)                                
                model.C = C_array(cc);
                [model_small, loo_err, loo_pred] = kls_train_zita_K(model,1);
                
                Ktest = utility_predict(cue, model_small);
                [pred, margins_zita] = k_predict_K(Ktest, model_small);
                err_tmp(cc) = numel(find(sign(margins_zita)==cue{1}.t_test'))/numel(margins_zita);
            end
            [V,I] = max(err_tmp);
            no_acc(i,s,j) = err_tmp(I);
            clear model_small
            
            clear err_tmp
            % Prior-Features
            pr_tr = prp(idx_tr,:);
            pr_te = prp(idx_te,:);
            
            for cc=1:numel(C_array)
                model_pr = linear_train(Y,sparse(pr_tr),['-B 1 -s 4 -c ',num2str(C_array(cc))]);
                [yhat, acc, dec] = linear_predict(Ytest,sparse(pr_te),model_pr);
                err_tmp(cc) = acc(1);
                clear model_pr
            end
            [V,I] = max(err_tmp);
            pr_acc(i,s,j) = err_tmp(I);
                        
            clear err_tmp

            % H-L2L(SVM-DAS)
            o_tr = orp(idx_tr,:);
            o_te = orp(idx_te,:);            
            
            for cc=1:numel(C_array)
                model_da = linear_train(Y,sparse(o_tr),['-B 1 -s 4 -c ',num2str(C_array(cc))]);
                [yhat, acc, dec] = linear_predict(Ytest,sparse(o_te),model_da);
                err_tmp(cc) = acc(1);
                clear model_da
            end
            [V,I] = max(err_tmp);
            da_acc(i,s,j) = err_tmp(I);
            
        end
    end
end

fprintf('done, saving file ''%s''\n',savefile);
save(savefile,'*_acc');
end

