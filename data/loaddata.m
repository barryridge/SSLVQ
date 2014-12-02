%% Clear...
% clear classes;

%% Load the data...
DataRootDir = '../../Data';
load([DataRootDir '/ICRA2009Project/PushCenter/features.mat']);

%% Convert it to CrossMod format...
Data = OPARC_CrossMod_data_struct_converter(Data);

%% Restrict the training class set...
Data.AllowedTrainingClassIndices = [1 2 3 4 5 7 11 13];
Data.GroundTruthClassIndices = [15 16];

%% Split the feature set into multiple modalities...
% Modality 1:
Data.Modalities{1}.FeatureIndices = [1:6 8:12];
Data.Modalities{1}.FeatureMask = zeros(1,24);
Data.Modalities{1}.FeatureMask([1:6 8:12]) = 1;

% Modality 2:
Data.Modalities{2}.FeatureIndices = [13:24];
Data.Modalities{2}.FeatureMask = zeros(1,24);
Data.Modalities{2}.FeatureMask([13:24]) = 1;