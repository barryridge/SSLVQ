RootPath = pwd;

% CrossMod source directory...
addpath([RootPath '/src']);

% Data directory...
addpath([RootPath '/data']);

% Experiments directory...
addpath([RootPath '/experiments']);

% Path to location of useful tools...
ToolsRootPath = [RootPath '/../../Tools'];

% Path to location of data sets...
DataRootPath = [RootPath '/../../Data'];

% Add the tools path...
addpath(ToolsRootPath);

% SOM Toolbox...
% http://www.cis.hut.fi/somtoolbox/
addpath([ToolsRootPath '/somtoolbox']);

% Earth Mover's distance MEX function...
% http://www.mathworks.com/matlabcentral/fileexchange/12936-emd-earth-mover
% s-distance-mex-interface
% http://ai.stanford.edu/~rubner/emd/default.htm
addpath([ToolsRootPath '/Metrics/emd']);

% LIBSVM...
% http://www.csie.ntu.edu.tw/~cjlin/libsvm/
% addpath([ToolsRootPath '/SVM/libsvm-mat-3.0-1']);
% addpath([ToolsRootPath '/SVM/libsvm-3.12/matlab']);
addpath([ToolsRootPath '/SVM/libsvm-3.20/matlab']);

% Multi-Parametric Toolbox
% http://control.ee.ethz.ch/~mpt/
addpath(genpath([ToolsRootPath '/mpt']));

% Decent bar plots...
% http://www.mathworks.com/matlabcentral/fileexchange/10803-barweb-bargraph-with-error-bars
addpath([ToolsRootPath '/barweb']);

% Export figures...
% http://www.mathworks.com/matlabcentral/fileexchange/23629-exportfig
addpath([ToolsRootPath '/Figures/export_fig']);
% http://www.mathworks.com/matlabcentral/fileexchange/7401-scalable-vector-
% graphics-svg-export-of-figures
addpath([ToolsRootPath '/Figures/plot2svg']);

% Statistics...
addpath([ToolsRootPath '/Statistics']);

% Neural Networks by Sebastien Paris from the Matlab File Exchange...
% http://www.mathworks.com/matlabcentral/fileexchange/17415-neural-network-classifiers
addpath([ToolsRootPath '/Neural Network Classifiers/NN']);

% X-Means Clustering by Dan Pelleg
% http://www.cs.cmu.edu/~dpelleg
addpath([ToolsRootPath '/Clustering/X-Means/auton']);

% Fast K-Means by Tim J. Benham
% [1] "k-means++: The Advantages of Careful Seeding", by David Arthur and
% Sergei Vassilvitskii, SODA 2007.
addpath([ToolsRootPath '/Clustering/fkmeans']);

% Random Forests
% https://code.google.com/p/randomforest-matlab/
addpath([ToolsRootPath '/Classification/randomforest-matlab/RF_Class_C']);

% Random Forest by Leo
% http://www.mathworks.com/matlabcentral/fileexchange/31036-random-forest
addpath([ToolsRootPath '/Stochastic_Bosque']);
