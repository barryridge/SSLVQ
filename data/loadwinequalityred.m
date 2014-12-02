Foo = load('./wine-quality/winequality-red.csv', '-ascii');
Y = Foo(:,end);
X = Foo(:,1:end-1);

Data.FeatureNames{1} = 'fixed acidity';
Data.FeatureNames{2} = 'volatile acidity';
Data.FeatureNames{3} = 'citric acid';
Data.FeatureNames{4} = 'residual sugar';
Data.FeatureNames{5} = 'chlorides';
Data.FeatureNames{6} = 'free sulfur dioxide';
Data.FeatureNames{7} = 'total sulfur dioxide';
Data.FeatureNames{8} = 'density';
Data.FeatureNames{9} = 'pH';
Data.FeatureNames{10} = 'sulphates';
Data.FeatureNames{11} = 'alcohol';

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