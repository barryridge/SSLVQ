Foo = load('./wine/wine.data', '-ascii');
Y = Foo(:,1);
X = Foo(:,2:end);

Data.FeatureNames{1} = 'Alcohol';
Data.FeatureNames{2} = 'Malic acid';
Data.FeatureNames{3} = 'Ash';
Data.FeatureNames{4} = 'Alcalinity of ash';
Data.FeatureNames{5} = 'Magnesium';
Data.FeatureNames{6} = 'Total phenols';
Data.FeatureNames{7} = 'Flavanoids';
Data.FeatureNames{8} = 'Nonflavanoid phenols';
Data.FeatureNames{9} = 'Proanthocyanins';
Data.FeatureNames{10} = 'Color intensity';
Data.FeatureNames{11} = 'Hue';
Data.FeatureNames{12} = 'OD280/OD315 of diluted wines';
Data.FeatureNames{13} = 'Proline';

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