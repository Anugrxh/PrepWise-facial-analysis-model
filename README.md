# Facial Emotion Analysis API

A fast and optimized Django REST API for real-time facial emotion detection from video files using deep learning.

## Features

- **Fast Processing**: Optimized frame sampling and batch processing
- **Emotion Detection**: Detects 7 emotions (neutral, happy, sad, angry, fear, surprise, disgust)
- **Performance Metrics**: Provides confidence, eye contact, speech clarity scores
- **RESTful API**: Easy integration with any frontend application
- **Memory Efficient**: Processes videos in chunks to handle large files

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Migrations**
   ```bash
   python manage.py migrate
   ```

3. **Start Development Server**
   ```bash
   python manage.py runserver
   ```

## API Usage

### Endpoint: `/api/facial-analysis/`

**Method:** POST  
**Content-Type:** multipart/form-data

**Request:**
- `video`: Video file (MP4, AVI, MOV supported)

**Response Example:**
```json
{
  "facialAnalysisResult": {
    "confidence": 75,
    "emotions": {
      "neutral": 60,
      "happy": 25,
      "fear": 10,
      "sad": 3,
      "angry": 1,
      "surprise": 1,
      "disgust": 0
    },
    "eyeContact": 68,
    "speechClarity": 72,
    "overallScore": 71,
    "feedback": "Good overall presentation with room for improvement in confidence"
  }
}
```

## Performance Optimizations

1. **Model Singleton**: Emotion detection model is loaded once and reused
2. **Frame Sampling**: Processes every 5th frame for faster analysis
3. **Batch Processing**: Analyzes multiple frames together
4. **Memory Management**: Automatic cleanup of temporary files
5. **Concurrent Processing**: Uses threading for better performance

## Testing

Use the provided test script:
```bash
python test_api.py
```

Make sure to replace `sample_video.mp4` with your actual video file path.

## Model Requirements

Ensure your `emotion_detction_model.h5` file is in the project root directory. The model should:
- Accept input shape: (48, 48, 1) grayscale images
- Output 7 emotion probabilities in order: [angry, disgust, fear, happy, sad, surprise, neutral]

## Error Handling

The API handles various error cases:
- Missing video file
- Corrupted video files
- No face detected in video
- Model loading errors
- Memory limitations

## Production Considerations

For production deployment:
1. Set `DEBUG = False` in settings.py
2. Configure proper CORS settings
3. Use a production WSGI server (gunicorn, uwsgi)
4. Add proper logging and monitoring
5. Implement rate limiting
6. Use cloud storage for video processing