from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import librosa as lb
import torch


model_name = "facebook/wav2vec2-large-xlsr-53-spanish"


# Initialize the tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# Initialize the model
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Read the sound file
waveform, rate = lb.load('./test.wav', sr = 16000)

# Tokenize the waveform
input_values = tokenizer(waveform, return_tensors='pt').input_values

# Retrieve logits from the model
logits = model(input_values).logits

# Take argmax value and decode into transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

# Print the output
print(transcription)