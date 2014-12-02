Foo = load('./soybean/soybean-small-numeric.data', '-ascii');
Y = Foo(:,end);
X = Foo(:,1:end-1);

for iFeature = 1:size(X,2)
    Data.FeatureNames{iFeature} = ['Feature ' num2str(iFeature)];
end

for iY = 1:size(Y,1)
    StrY{iY} = num2str(Y(iY));
end
Y = StrY;

Data.ClassNames = unique(Y);

Data.FeatureVectors = X';

for iClass = 1:size(Data.ClassNames,2)
    Data.ClassLabels(iClass,:) = strcmp(Y, Data.ClassNames{iClass})';
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
clear X;
clear Y;
clear Foo;
clear Bar;
clear StrY;
clear iY;