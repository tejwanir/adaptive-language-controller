# Installation

1. Install pip dependencies: `pip install -r requirements.txt`

2. Install `ffmpeg` for Whisper transcription: `brew install ffmpeg`

# Models

* model_phrase_v0:
* model_phrase_v1: trained without ignore_index=0, max_text_seq_length=64
* model_phrase_v1.1: trained wihtout ignore_index=0, max_text_seq_length=32
    * num_epochs=20, lr=5e-4
    * always starts with "keep"
* model_phrase_v1.2: trained with ignore_index=0, all joints
    * num_epochs=30, lr=5e-4
    * lots of repetition of "body" "your"
* model_phrase_v1.3: train with ignore_index=0, only WristLeft
    * num_epochs=20, lr=5e-4
    * very little diversity in predictions