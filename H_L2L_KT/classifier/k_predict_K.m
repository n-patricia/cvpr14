function [predicted_label,margins] = k_predict(K,model,average)
% K_PREDICT Generic prediction function
%    [predicted_label,margins] = k_predict(x,model)

if nargin<3
    average=0;
end

if iscell(model)==1
    %[predicted_label,margins] = k_predict_multi(x,model,average);
    disp('multiclass');
    return
end

nt = size(K,1); %deve essere il num dei campioni di test
max_num_el=50*1024^2/8; %50 Mega of memory as maximum size for K
step=ceil(max_num_el/numel(model.beta));
for i=1:step:nt
    if average==0
        margins(:,i:min(i+step-1,nt)) = (K*model.beta+model.b)';
    else
        margins(:,i:min(i+step-1,nt)) = (K*model.beta2+model.b)';
    end
end

if size(model.beta,2)>1
    [tmp,predicted_label]=max(margins);
else
    [predicted_label]=sign(margins);
end

%elseif nargout>1
%    Kbb=feval(model.ker,model.X,model.S,model.S,model.kerparam);
%    for i=1:size(x,2)
%        out_var(i) = feval(model.ker,x,i,i,model.kerparam);
%    end
%    out_var=out_var+1/model.C-diag(K*pinv(Kbb+1/model.C)*K')';
%end