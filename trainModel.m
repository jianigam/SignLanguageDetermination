% Load the dataset
imds = imageDatastore('C:\Users\HP\Desktop\SignLanguageToSpeech\Dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Resize all images
inputSize = [64 64];
imds.ReadFcn = @(filename) imresize(imread(filename), inputSize);

% Exclude J, N, and P
excludedLabels = ["J", "N", "P"];
imds = subset(imds, ~ismember(imds.Labels, excludedLabels));

% Get remaining labels (A–Z excluding J, N, P)
labels = unique(imds.Labels);

% Balance dataset: ~435 samples per class to make ~10,000 total
minSamplesPerClass = 435;
balancedFiles = [];

for i = 1:numel(labels)
    label = labels(i);
    idx = find(imds.Labels == label);
    selectedIdx = idx(randperm(length(idx), min(minSamplesPerClass, length(idx))));
    balancedFiles = [balancedFiles; selectedIdx];
end

imdsBalanced = subset(imds, balancedFiles);

% Split into training (90%) and validation (10%)
[imdsTrain, imdsValidation] = splitEachLabel(imdsBalanced, 0.9, 'randomized');

% Define CNN architecture
layers = [
    imageInputLayer([64 64 3], 'Name', 'input')

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(numel(labels)) % 23 classes
    softmaxLayer
    classificationLayer
];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 16, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu', ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 0.001);

% Train the network
net = trainNetwork(imdsTrain, layers, options);

% Evaluate on validation set
YPred = classify(net, imdsValidation);
YVal = imdsValidation.Labels;
accuracy = sum(YPred == YVal) / numel(YVal);
disp("Validation Accuracy: " + accuracy);
