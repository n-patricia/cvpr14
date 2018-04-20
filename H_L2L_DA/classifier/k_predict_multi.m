function [out,vals] = k_predict_multi(x,models,average)
% K_PREDICT_MULTI Generic prediction for multiclass models
%    [out,vals] = k_predict_multi(models,X)

if nargin<3
    average=0;
end

for i=1:numel(models)
    [dummy,vals(i,:)] = k_predict_(x,models{i},average);
end

[tmp,out]=max(vals);