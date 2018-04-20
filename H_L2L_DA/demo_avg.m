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

clear

sources = {'amazon'};
target = 'webcam';

create_prior_avg(sources, target);
hl2l_da(sources, target);