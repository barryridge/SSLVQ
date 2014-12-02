%% Clear...
% clear classes;

%% Load the data...
% DataRootDir = '../../../Data';
% load([DataRootDir '/ICRA2009Project/PushCenter/features2.mat']);
load([DataRootPath '/ICRA2009Project/PushCenter/features2.mat']);

%% Convert it to CrossMod format...
Data = OPARC_CrossMod_data_struct_converter(Data);

%% Restrict the training class set...
Data.AllowedTrainingClassIndices = [1 2 3 4 5 7 11 13];
Data.GroundTruthClassIndices = [14 15];

%% Split the feature set into multiple modalities...
% Modality 1:
Data.Modalities{1}.FeatureIndices = [1:36];
% Data.Modalities{1}.FeatureIndices = [1 2 5 6 9 10 13 14 17 18 21 22 25 26 29 30 33 34];
% Data.Modalities{1}.FeatureIndices = [3 4 7 8 11 12 15 16 19 20 23 24 27 28 31 32 35 36];
Data.Modalities{1}.FeatureMask = zeros(1,length(Data.FeatureNames));
Data.Modalities{1}.FeatureMask([1:36]) = 1;

% Modality 2:
Data.Modalities{2}.FeatureIndices = [length(Data.FeatureNames)-11:length(Data.FeatureNames)];
Data.Modalities{2}.FeatureMask = zeros(1,length(Data.FeatureNames));
Data.Modalities{2}.FeatureMask([length(Data.FeatureNames)-11:length(Data.FeatureNames)]) = 1;
