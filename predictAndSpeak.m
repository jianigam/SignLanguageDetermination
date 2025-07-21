clc;
clear;
close all;

% Load the trained model
load(fullfile('TrainedModel', 'asl_model.mat'), 'net');

% Define input size expected by CNN
inputSize = net.Layers(1).InputSize;

% Start webcam and set to a supported resolution
cam = webcam;
cam.Resolution = '640x480';  % You can try '320x240' if it's still slow

% Ensure webcam is released when script ends
cleanupObj = onCleanup(@() clear('cam')); 


disp('Press Ctrl+C to stop...');

while true
    % Capture a frame
    img = snapshot(cam);

    % Convert to grayscale if CNN expects single channel input
    if inputSize(3) == 1
        img = rgb2gray(img);
    end

    % Resize image to match model input size
    resizedImg = imresize(img, inputSize(1:2));

    % If grayscale, reshape to include the 3rd dimension
    if inputSize(3) == 1
        resizedImg = reshape(resizedImg, inputSize);
    end

    % Predict label
    label = classify(net, resizedImg);

    % Show image and prediction
    imshow(img);
    title(['Predicted Sign: ', char(label)], 'FontSize', 16);

    % Speak out the predicted label
    textToSpeech(char(label));

    pause(2);  % wait before next prediction
end

% Text-to-Speech function
function textToSpeech(text)
    if ispc
        NET.addAssembly('System.Speech');
        speaker = System.Speech.Synthesis.SpeechSynthesizer;
        speaker.Speak(text);
    else
        disp(['Say: ', text]);
    end
end