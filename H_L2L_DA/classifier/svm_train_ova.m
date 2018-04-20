% train a one vs. all SVM using libsvm 
function model = svm_train_ova(labels, matrix, options)


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


if numel(find(labels<0))>0
  error('All labels must be integers larger than 0')
end

classes = unique(labels);
model.ovaModels = cell(numel(classes), 1);

for i=1:numel(classes)
  cls_labels = labels;
  idx = find(labels ~= classes(i)); 
  cls_labels(idx) = -1;
  model.ovaModels{i}.libsvm = svm_train(cls_labels, matrix, options);
  model.ovaModels{i}.cls = classes(i);
end

SVs = [];
for i=1:numel(classes)
  SVs = [SVs; full(model.ovaModels{i}.libsvm.SVs)];
end
SVs = unique(SVs, 'rows');
model.nTotalSVs  = size(SVs, 1);
