Foo = load('./pima-indians-diabetes/pima-indians-diabetes.data', '-ascii');
Y = Foo(:,end);
X = Foo(:,1:end-1);

Data.FeatureNames{1} = 'Number of times pregnant';
Data.FeatureNames{2} = 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test';
Data.FeatureNames{3} = 'Diastolic blood pressure (mm Hg)';
Data.FeatureNames{4} = 'Triceps skin fold thickness (mm)';
Data.FeatureNames{5} = '2-Hour serum insulin (mu U/ml)';
Data.FeatureNames{6} = 'Body mass index (weight in kg/(height in m)^2)';
Data.FeatureNames{7} = 'Diabetes pedigree function';
Data.FeatureNames{8} = 'Age (years)';

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