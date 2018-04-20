% test a one vs all SVM using libsvm
function [pred, acc, dec ] = svm_predict_ova(labels, matrix, model, options)


% This code is part of the supplementary material to the ICCV 2011 paper
% "Multiclass Transfer Learning from Unconstrained Priors", L. Jie*, T.
% Tommasi*, B. Caputo (* equal authorship). 

% Copyright (C) 2010-2011, Tatiana Tommasi
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
% Contact the authors: ttommasi [at] idiap.ch , jluo [at] idiap.ch


if ~iscell(model.ovaModels)
  error('Incorrect One vs. All SVM model');
end

if nargin < 4
 options = [];   
end

n_cls    = numel(model.ovaModels);
n_sample = numel(labels);

classes = zeros(n_cls, 1);
dec  = zeros(n_sample, n_cls);

for i=1:n_cls
  classes(i) = model.ovaModels{i}.cls;
  cls_labels = labels;
  idx = find(labels ~= model.ovaModels{i}.cls); 
  cls_labels(idx) = -1;

  fprintf('Testing model of class %d\n', model.ovaModels{i}.cls);
  [tmppred tmpacc tmpdec] = svm_predict(cls_labels, matrix, model.ovaModels{i}.libsvm, options);
  fprintf('\n');
  if model.ovaModels{i}.libsvm.Label(1) == -1
    dec(:, i) = -tmpdec;
  else
    dec(:, i) = tmpdec;
  end
end

[ maxdec idx ] = max(dec, [], 2);

pred = classes(idx);
acc(1)=numel(find(pred==labels))/n_sample;

acc_cls = zeros(n_cls,1);
for i=1:n_cls
  idx = find(labels == classes(i));
  if ~isempty(idx)  
   acc_cls(i) = numel(find(pred(idx) == labels(idx)))/numel(idx);
  else
   acc_cls(i) = 1;   
  end
end
acc(2) = mean(acc_cls);

fprintf('Overall accuracy = %.4f%% (%d/%d)\n', acc(1)*100, numel(find(pred==labels)), n_sample);
fprintf('Average accuracy per category = %.4f%%\n', acc(2)*100);

