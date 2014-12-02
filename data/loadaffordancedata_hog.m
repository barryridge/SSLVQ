%% Clear...
% clear classes;

%% Load the data...
DataRootDir = '../../../Data';
load([DataRootDir '/ICRA2009Project/PushCenter/features2.mat']);

%% Convert it to CrossMod format...
Data = OPARC_CrossMod_data_struct_converter(Data);

%% Restrict the training class set...
Data.AllowedTrainingClassIndices = [1 2 3 4 5 7 11 13];
Data.GroundTruthClassIndices = [15 16];

%% Split the feature set into multiple modalities...
% Modality 1:
Data.Modalities{1}.FeatureIndices = [84:137];
Data.Modalities{1}.FeatureMask = zeros(1,length(Data.FeatureNames));
Data.Modalities{1}.FeatureMask([84:137]) = 1;

% Modality 2:
Data.Modalities{2}.FeatureIndices = [length(Data.FeatureNames)-11:length(Data.FeatureNames)];
Data.Modalities{2}.FeatureMask = zeros(1,length(Data.FeatureNames));
Data.Modalities{2}.FeatureMask([length(Data.FeatureNames)-11:length(Data.FeatureNames)]) = 1;
