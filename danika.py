import streamlit as st
from pydub import AudioSegment
import librosa
import numpy as np
import pandas as pd
import requests
import json

#functions
#audio processing function
def librosa_feature_extractor(src_path):
    """
    @ param src_path: string: path to the audio file
    @ returns features: dictionary: keys are feature names, values are the value
                                    of each feature
    """
    # Convert song to features
    y, sr = librosa.load(src_path, mono=True, duration=30)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if type(tempo) == int:
        tempo = np.float32(tempo)

    print(tempo.item())
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    features = {"tempo": tempo.item(),
                "beats": beats.shape[0], "chroma_stft":np.mean(chroma_stft).item(), "rmse": np.mean(rmse).item(), 
                "spectral_centroid": np.mean(spec_cent).item(),
                "spectral_bandwidth": np.mean(spec_bw).item(),
                "rolloff": np.mean(rolloff).item(),
                "zero_crossing_rate": np.mean(zcr).item()}

    iterator = 1
    for e in mfcc:
        feature_name = "mfcc" + str(iterator)
        features[feature_name]=np.mean(e).item()
        iterator += 1

    return features


def get_prediction(data={"tempo":165.83041487,"rmse":0.019552287327550003,"beats":64,"chroma_stft":0.4692273885,"rolloff":5012.7839675000005,"spectral_centroid":2385.7171495000002,"spectral_bandwidth":2073.4562841,"mfcc2":136.261322015,"zero_crossing_rate":0.14890140496,"mfcc1":-673.40856935,"mfcc5":1.0822525049999978,"mfcc3":-37.75925637499999,"mfcc4":11.574878695,"mfcc8":7.089370730000001,"mfcc6":-10.745647429999998,"mfcc7":-3.7076854700000013,"mfcc11":-5.119864700500001,"mfcc9":-10.045778274999998,"mfcc10":1.3571209899999994,"mfcc14":1.573728085,"mfcc12":-7.446875575,"mfcc13":0.548441884999999,"mfcc17":-0.9281492199999999,"mfcc15":4.199724675000001,"mfcc16":3.179331305,"mfcc20":3.1866111750000004,"mfcc18":4.820736885,"mfcc19":0.054489134999999855}):
  url = 'https://askai.aiclub.world/c98818c0-0c4b-4ce8-a42a-805dee8197a7'
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")
  print(response)
  return response

#web app
#project title
#sample project is created here in a way this part can be extracted
st.title("Audio Classification")

#input file
audio_file = st.file_uploader("Upload an audio file", accept_multiple_files=False, help="Upload either .wav or mp3", type=["wav","mp3"])

if audio_file:
    audio = AudioSegment.from_wav(audio_file)
    audio.export("input_audio.wav", format = "wav")
    print("Audio file is saved")

    #process the audio file
    audio_path = "input_audio.wav"
    features = librosa_feature_extractor(audio_path)
    #show the results
    st.subheader("Audio Preprocessing Results")
    st.dataframe(pd.DataFrame(features, index = [0]))
    response = get_prediction(features)
    response = json.loads(json.loads(response)['body'])['predicted_label']
    st.subheader("Mosquito is: {}".format(response))
