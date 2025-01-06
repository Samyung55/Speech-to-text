from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
import torch
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load the Whisper model and processor
model_name = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """
    Transcribe audio using the Whisper model.
    """
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.info(f"Loaded audio file: {audio_path}, sample rate: {sample_rate} Hz")

        # Resample the audio to 16kHz if needed (Whisper expects 16kHz audio)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            logger.info(f"Resampled audio to 16kHz")

        # Preprocess the audio
        inputs = processor(
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

        # Decode the output to text
        transcription = processor.decode(output[0], skip_special_tokens=True)
        logger.info(f"Transcription successful: {transcription}")
        return transcription

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
@app.post("/transcribe/")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload an audio file and get its transcription.
    """
    # Log the file details
    logger.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    # Validate the file
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Save the uploaded file temporarily
    temp_audio_path = "temp_audio.mp3"
    try:
        with open(temp_audio_path, "wb") as f:
            content = await file.read()
            logger.info(f"File size: {len(content)} bytes")
            f.write(content)

        # Transcribe the audio
        transcription = transcribe_audio(temp_audio_path, language="en")

        # Clean up the temporary file
        os.remove(temp_audio_path)
        logger.info(f"Deleted temporary file: {temp_audio_path}")

        return {"transcription": transcription}

    except Exception as e:
        # Clean up the temporary file if an error occurs
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.info(f"Deleted temporary file after error: {temp_audio_path}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)