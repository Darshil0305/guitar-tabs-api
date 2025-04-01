from flask import Blueprint, request, jsonify
from .services.tab_generator import generate_tabs_from_youtube
import traceback

main_bp = Blueprint('main', __name__)

@main_bp.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"})

@main_bp.route('/api/generate-tabs', methods=['POST'])
def generate_tabs():
    """
    Generate guitar tabs from a YouTube URL
    Expected JSON body:
    {
        "url": "YouTube URL",
        "use_capo": boolean,
        "is_fingerstyle": boolean
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({"error": "Missing YouTube URL"}), 400
            
        youtube_url = data.get('url')
        use_capo = data.get('use_capo', False)
        is_fingerstyle = data.get('is_fingerstyle', False)
        
        # Call the tab generation service
        tabs = generate_tabs_from_youtube(youtube_url, use_capo, is_fingerstyle)
        
        return jsonify({
            "tabs": tabs,
            "song_details": {
                "title": tabs.get("title", "Unknown Song"),
                "artist": tabs.get("artist", "Unknown Artist"),
                "videoId": tabs.get("video_id", "")
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500 