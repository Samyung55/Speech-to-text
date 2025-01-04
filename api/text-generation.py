from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForSpeechSeq2Seq, AutoFeatureExtractor, AutoTokenizer
import torchaudio
import torch
import os

# Initialize FastAPI app
app = FastAPI()

# Load the model, feature extractor, and tokenizer
model_path = "./speech_to_text/"  # Update this path if needed
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def transcribe_audio(audio_path, language="en"):
    """
    Transcribe audio using the model, feature extractor, and tokenizer.
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample the audio to the required sample rate (if needed)
    # Whisper models expect 16kHz audio
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Preprocess the audio
    inputs = feature_extractor(
        waveform.squeeze().numpy(), 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    )
    
    # Generate transcription
    with torch.no_grad():
        output = model.generate(
            inputs["input_features"],
            language=language,  # Force transcription in English
            task="transcribe"   # Explicitly set the task to transcription
        )
    
    # Decode the output to text using the tokenizer
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

@app.post("/transcribe/")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload an audio file and get its transcription.
    """
    # Save the uploaded file temporarily
    temp_audio_path = "temp_audio.mp3"
    try:
        with open(temp_audio_path, "wb") as f:
            f.write(await file.read())
        
        # Transcribe the audio
        transcription = transcribe_audio(temp_audio_path, language="en")
        
        # Clean up the temporary file
        os.remove(temp_audio_path)
        
        return {"transcription": transcription}
    
    except Exception as e:
        # Clean up the temporary file if an error occurs
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)