# Guitar Tabs API

A Python Flask API for generating guitar tabs from YouTube videos.

## Features

- Extract audio from YouTube videos
- Analyze audio to detect notes and rhythms
- Generate guitar tabs with options for capo and playing style
- Return formatted tab notation

## Prerequisites

- Python 3.8+
- FFmpeg (required for audio processing)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd guitar-tabs-api
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Make sure FFmpeg is installed and available in your PATH.

## Running the API

Start the development server:
```
python run.py
```

The API will be available at http://localhost:5000.

## API Endpoints

### `POST /api/generate-tabs`

Generate guitar tabs from a YouTube URL.

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "use_capo": true,
  "is_fingerstyle": false
}
```

**Response:**
```json
{
  "tabs": {
    "title": "Song Title",
    "artist": "Artist Name",
    "video_id": "dQw4w9WgXcQ",
    "tab_content": "E|------|\nB|------|\nG|--2---|\nD|------|\nA|------|\nE|--0---|\n\nCapo on fret 2\nStrumming pattern",
    "use_capo": true,
    "is_fingerstyle": false
  },
  "song_details": {
    "title": "Song Title",
    "artist": "Artist Name",
    "videoId": "dQw4w9WgXcQ"
  }
}
```

### `GET /api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Limitations

This is a basic implementation with several limitations:

1. Audio analysis is rudimentary and may not always produce accurate tabs
2. Complex musical techniques (bends, slides, etc.) are not detected
3. Requires FFmpeg for audio processing
4. YouTube terms of service should be respected when using this API

## License

MIT 