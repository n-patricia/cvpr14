function create_sourceout( dataset )
%CREATE_SOURCEOUT Summary of this function goes here
% create source output confidence for binary transfer learning
% INPUT: dataset
% OUTPUT: source confidence file

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

load(dataset);
n = numel(names);
savefile = strrep(dataset,'.mat','_source_out.mat');

load_settings;

fprintf('computing source output confidence...\n');
for i=1:n
    fprintf('SUBJ:%d\n',i);
    for ii=1:n
        if ii==i, continue; end
        
        for f=1:numel(feat)
        p_tr = p{ii,f};
        t_tr = t{ii,f};
        nY = length(t_tr');
        
        hp.type = 'rbf';
        hp.gamma = 1/mean2(dist_euclid(p_tr,p_tr));    
                
        for cc=1:numel(C_array)
            
            model_init =  k_init(@compute_kernel, hp, C_array(cc));

            Ktrain = feval(model_init.ker, p_tr', 1:nY, 1:nY, model_init.kerparam);
            model_init.X = p_tr;
            model_init.K = Ktrain;
            model_init.Y = t_tr';
            model_init.S = [1:nY];
            [model_loo{ii,f,cc}, loo_err, loo_pred] = kls_train_K(model_init,0);
            
            clear Ktrain model_init
        end
        end
    end
    
    
    for ii=1:n
    if ii==i, continue; end
    clear err_tmp out_conf
    for f=1:numel(feat)
        cue.p_test = p{i,f};
        cue.t_test = t{i,f};

        for cc=1:numel(C_array)
            out_conf{cc} = zeros(numel(t{i,1}),1);
            Ktest = utility_predict_single(cue, model_loo{ii,f,cc});
            [pred,vals] = k_predict_K(Ktest, model_loo{ii,f,cc});
            err_tmp(cc) = numel(find(pred==t{i,f}'))/numel(t{i,f});
            out_conf{cc} = vals';
            clear Ktest
        end
        
        [V,I] = max(err_tmp);
        s_out_conf{i,ii}(:,f) = out_conf{I};
        clear cue
    end

    
    end
    clear model_loo
end

fprintf('done, saving ''%s''\n\n',savefile);
save(savefile,'s_out_conf');

end
