""" 
-----------------------------------------------------------------------------------------
Proj3_live_inference_2.py
Michael J Evan
CIS 602 Fundamentals of Deep Learning
University of Massachussets Dartmouth
Graduate Computer Science Department
Fall 2025

Project 3 RNN (Recurrent Neural Networks)

Utilizing PyTorch & Librosa for real-time audio classification
GRU & LSTM PyTorch Deep Learning Models

Note: RUN VIA TERMINAL IN ACTIVE VENV WITH ALL REQUIRED LIBRARIES !!!

* This script performs live audio inference using pre-trained GRU and LSTM models.
* Voice to text conversion in real-time

* Current list of recognized commands:
* "bird", "cat", "dog", "down", "forward", "go", "left", "no", "right", "stop", "up", "yes"
-------------------------------------------------------------------------------------------
"""

''' Note: Run via terminal to avoid downloads for imports'''
import time
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn

'''----------------- User settings -----------------'''
MODEL_PATH = "best_gru.pt"      # best_gru.pt or best_lstm.pt 
MODEL_TYPE = "gru"              # "gru" or "lstm"
LABELS = []                     # left empty -> will be replaced by checkpoint labels if present

SR = 16000                      # sampling rate used in training
N_MELS = 40                     # mel filter count (must match training)
WIN = 400                       # n_fft (~25ms at 16k)
HOP = 160                       # hop length (~10ms at 16k)
FRAMES = 100                    # number of frames expected (~1s window)

# VAD / buffer / thresholds
WINDOW_SEC = 1.0                # 4 blocks / inference window if BLOCK_SEC=0.25
WINDOW = int(SR * WINDOW_SEC)   # rolling buffer size in samples
BLOCK_SEC = 0.25                # process audio every 250ms
BLOCK = int(SR * BLOCK_SEC)     # input callback block size (samples/call)
VAD_RMS_THRESHOLD = 0.1         # RMS threshold for voice activity
COOLDOWN = 1.0                  # seconds between triggers
CONF_THRESH = 0.70              # confidence threshold for printed recognition
'''--------------------------------------------------'''

''' Model classes (must match attribute names used when saving checkpoint)'''
class GRUModel(nn.Module):
    def __init__(self, in_dim, hid, nl, nc):
        super().__init__()
        # attribute names 'rnn' and 'fc' must match checkpoint keys
        self.rnn = nn.GRU(in_dim, hid, nl, batch_first=True)    # GRU layer
        self.fc  = nn.Linear(hid, nc)   # fully connected layer
    def forward(self, x):
        out, _ = self.rnn(x)            # out shape: (B, T, hidden)
        return self.fc(out[:, -1, :])   # use last time step

class LSTMModel(nn.Module):
    def __init__(self, in_dim, hid, nl, nc):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hid, nl, batch_first=True)
        self.fc  = nn.Linear(hid, nc)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
'''------------------------------------------------------------------'''

# Load checkpoint (safe CPU load), restore labels if saved, instantiate model
ck = torch.load(MODEL_PATH, map_location="cpu")
LABELS = ck.get("labels", LABELS) or LABELS # restore saved label order if present
num_classes = len(LABELS)                   # number of classes
# pick the correct model class
model = (GRUModel(N_MELS, 128, 2, num_classes) if MODEL_TYPE.lower().startswith("g")
         else LSTMModel(N_MELS, 128, 2, num_classes)) # instantiate the model

state = ck.get("model_state", ck)           # support dict or raw state dict
model.load_state_dict(state)                # load weights (must match attribute names)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")  # select device
model.to(DEVICE)                            # move model to device BEFORE inference
model.eval()                                # set eval mode (disable dropout, etc.)

# Rolling buffer and control vars
buffer = np.zeros(WINDOW, dtype=np.float32) # rolling buffer for audio samples
active_count = 0                            # count of active blocks
last_trigger = 0.0                          # time of last trigger

# Preprocessing: compute mel spectrogram exactly as training
def make_mel(y):
    # mono, resample not needed here (input from device at SR)
    if y.ndim > 1:                          # average channels if multi-channel
        y = y.mean(axis=1)                  # average channels
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=WIN, hop_length=HOP,
                                       n_mels=N_MELS, power=2.0)    # compute mel spectrogram
    S_db = librosa.power_to_db(S, ref=np.max).T                     # (frames, n_mels)
    # pad/truncate to fixed FRAMES
    if S_db.shape[0] < FRAMES:              # pad if too short
        pad = np.zeros((FRAMES - S_db.shape[0], N_MELS), dtype=np.float32)
        S_db = np.vstack([S_db, pad])       # pad with zeros
    else:
        S_db = S_db[:FRAMES, :]             # truncate if too long
    # simple normalization used in training
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    return S_db.astype(np.float32)   

# audio callback: updates buffer, performs simple VAD, runs inference on trigger
def callback(indata, frames, time_info, status):
    global buffer, active_count, last_trigger
    if status:
        print("Audio status:", status)  
    data = indata[:, 0].astype(np.float32)   # mono channel
    n = len(data)
    # update rolling buffer (keep last WINDOW samples)
    if n >= WINDOW:
        buffer = data[-WINDOW:]
    else:
        buffer = np.roll(buffer, -n)
        buffer[-n:] = data

    # simple VAD using RMS of current block
    rms = float(np.sqrt(np.mean(data ** 2))) # root mean square
    now = time.time()                        # current time
    if rms >= VAD_RMS_THRESHOLD:             # voice activity detected
        active_count += 1                    # increment active block count
    else:
        active_count = 0                     # reset active block count

    # trigger: enough active blocks and cooldown passed
    if active_count >= 1 and (now - last_trigger) >= COOLDOWN:
        mel = make_mel(buffer)               # preprocess last WINDOW seconds
        x = torch.from_numpy(mel).unsqueeze(0).to(DEVICE).float()  # shape (1, T, F)
        with torch.no_grad():                # disable gradient calculation
            logits = model(x)                # forward pass
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]   # probabilities
        idx = int(probs.argmax())            # index of highest probability
        p = float(probs[idx])                # probability of highest class
        ts = time.strftime("%H:%M:%S")       # current time string
        if p >= CONF_THRESH:                 # confidence threshold check
            print(f"{ts}  RECOGNIZED 😎: {LABELS[idx]} (probability = {p:.2f}%)"), print()
        else:
            print(f"{ts}  UNRECOGNIZED  🔥 (highest-probability) = {LABELS[idx]} probability = {p:.2f}%)"), print()
        last_trigger = now                   # update last trigger time
        active_count = 0                     # reset active block count

'''----- Main loop: open InputStream and run callback until KeyboarInterupt: Ctrl-C -----'''
if __name__ == "__main__":
    print("Listening — press Ctrl+C to stop")
    print()
    try:
        with sd.InputStream(channels=1, samplerate=SR, blocksize=BLOCK, callback=callback):   
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print()
        print("\nStopped ----- 🛑 ----- later dude!"), print()