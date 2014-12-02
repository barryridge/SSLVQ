function Data = castasregression(Data)
% Cast a classification problem as a regression problem by converting the
% class labels to feature vectors in a seperate modality.

    % Pad out each modality feature mask...
    % (these masks are not currently used by CrossMod, but could be useful
    % to have anyway)
    for iMod = 1:size(Data.Modalities,1)
        Data.Modalities{iMod}.FeatureMask(end+1:end+size(Data.ClassLabels,1)) =...
            zeros(1,size(Data.ClassLabels,1));
    end
        
    % Add a modality for the class labels...
    Data.Modalities{end+1}.FeatureIndices =...
        size(Data.FeatureVectors,1)+1 :...
        size(Data.FeatureVectors,1) + size(Data.ClassLabels,1);
    
    Data.Modalities{end}.FeatureMask =...
        zeros(1,size(Data.FeatureVectors,1) + size(Data.ClassLabels,1));
    Data.Modalities{end}.FeatureMask(Data.Modalities{end}.FeatureIndices) =...
        ones(1,size(Data.Modalities{end}.FeatureIndices,2));
    

    % Add the class labels to the feature vectors using 1-of-n encoding...
    Data.FeatureVectors = [Data.FeatureVectors' Data.ClassLabels']';
    
    % Give the new features names...
    for iClass = 1:size(Data.ClassLabels,1)
        Data.FeatureNames{end+1} = Data.ClassNames{iClass};
    end
        
end