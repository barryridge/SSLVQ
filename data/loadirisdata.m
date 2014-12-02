load fisheriris

for iFeature = 1:size(meas,2)
    Data.FeatureNames{iFeature} = ['Measurement ' num2str(iFeature)];
end

Data.ClassNames = unique(species)';

Data.FeatureVectors = meas';

for iClass = 1:size(Data.ClassNames,2)
    Data.ClassLabels(iClass,:) = strcmp(species, Data.ClassNames{iClass})';
end

%% Restrict the training class set...
Data.AllowedTrainingClassIndices = 1:size(Data.ClassNames,2);
Data.GroundTruthClassIndices = 1:size(Data.ClassNames,2);

%% Split the feature set into multiple modalities...
% Modality 1:
Data.Modalities{1}.FeatureIndices = 1:size(Data.FeatureVectors,1);
Data.Modalities{1}.FeatureMask = ones(1,size(Data.FeatureVectors,1));

clear iClass;
clear iFeature;
clear meas;
clear species;