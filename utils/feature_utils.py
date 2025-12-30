import librosa

def extract_mfcc(signal):
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=22050,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )
    return mfcc.T
