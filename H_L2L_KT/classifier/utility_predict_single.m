function [kernel] = utility_predict(cue, model)

p_test=cue.p_test;
t_test=cue.t_test;

nt = size(p_test,1);
max_num_el=50*1024^2/8; %50 Mega of memory as maximum size for K
step=ceil(max_num_el/numel(model.beta));
for s=1:step:nt
    if isfield(model,'SV')
        xtest=p_test';
        K(:,:) = feval(model.ker,xtest(:,s:min(s+step-1,nt)),model.SV,model.kerparam);
    else
        nr = length(model.Y);
        xtrain=model.X';
        xtest=p_test';
        K(:,:) = feval(model.ker,[xtrain , xtest(:,s:min(s+step-1,nt))],(1:numel(s:min(s+step-1,nt)))+nr,model.S,model.kerparam);
    end
end

kernel=K;