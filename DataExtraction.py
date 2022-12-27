# Music Genre Classifier Dataset
import os
import math
import json
import librosa
import warnings
warnings.filterwarnings("ignore")

sampleRate = 22050
duration = 30 # Seconds
samplesPerTrack = sampleRate * duration

dataDir = "Path to dataset"
jsonDir = "genreData.json"
categories = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def saveMFCC(dataPath, jsonPath, nMFCC = 13, nFFT = 2048, hopLen = 512, numSegments = 5):
    data = { # Store data
        "genreMap": categories, # Genre mapping
        "mfcc": [], # Training inputs
        "labels": [] # Training outputs
    }

    numSamplesPerSegment = samplesPerTrack / numSegments
    MFCCPerTrack = math.ceil(numSamplesPerSegment / hopLen)

    # Loop through genres
    for idx, category in enumerate(categories):
        print("Processing: " + category)
        path = os.path.join(dataDir, category)
        for musicFile in os.listdir(path):
            signal, sr = librosa.load(os.path.join(path, musicFile))
            for segNum in range(numSegments): # Split data into smaller segments for more data
                startSample = int(numSamplesPerSegment * segNum)
                finishSample = int(startSample + numSamplesPerSegment)
                mfcc = librosa.feature.mfcc( # MFCC generation using Librosa
                    signal[startSample:finishSample],
                    n_mfcc = nMFCC,
                    n_fft = nFFT,
                    hop_length = hopLen)
                mfcc = mfcc.T

                # Store MFCC for segment if of expected len
                if (len(mfcc) == MFCCPerTrack):
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(idx)
    
    with open(jsonPath, "w") as fp:
        json.dump(data, fp, indent = 4)

saveMFCC(dataDir, jsonDir, numSegments = 10)