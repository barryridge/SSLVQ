function NewData = OPARC_CrossMod_data_struct_converter(Data)

    NewData.FeatureNames = Data.FeatureNames;
    NewData.ClassNames = Data.CategoryNames;    
    NewData.FeatureVectors = Data.FeatureVectors;
    NewData.ClassLabels = Data.CategoryLabels;
    NewData.Modalities{1}.FeatureIndices = Data.ObjPropFeatureIndices;
    NewData.Modalities{2}.FeatureIndices = Data.ResultFeatureIndices;
    NewData.Modalities{1}.FeatureMask = Data.ObjPropFeatureMask;
    NewData.Modalities{2}.FeatureMask = Data.ResultFeatureMask;
    
    for i = 1:length(Data.FeatureNames)
        VarName = strcat('Data.FeatureNames{', num2str(i), '}');
        
        while iscell(eval(VarName))
            VarName = strcat(VarName, '{1}');
        end
            
        NewData.FeatureNames{i} = eval(VarName);
    end
    
end