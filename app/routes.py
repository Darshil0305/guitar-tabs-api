from flask import Blueprint, request, jsonify
from .services.tab_generator import generate_tabs_from_youtube
import traceback
import logging

# Set up logging
logger = logging.getLogger(__name__)

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
        "is_fingerstyle": boolean,
        "use_source_separation": boolean
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({"error": "Missing YouTube URL"}), 400
            
        youtube_url = data.get('url')
        use_capo = data.get('use_capo', False)
        is_fingerstyle = data.get('is_fingerstyle', False)
        use_source_separation = data.get('use_source_separation', True)  # Default to True
        
        logger.info(f"Generating tabs for URL: {youtube_url}")
        logger.info(f"Options: use_capo={use_capo}, is_fingerstyle={is_fingerstyle}, use_source_separation={use_source_separation}")
        
        # Call the tab generation service
        tabs = generate_tabs_from_youtube(
            youtube_url, 
            use_capo=use_capo, 
            is_fingerstyle=is_fingerstyle,
            use_source_separation=use_source_separation
        )
        
        return jsonify({
            "tabs": tabs,
            "song_details": {
                "title": tabs.get("title", "Unknown Song"),
                "artist": tabs.get("artist", "Unknown Artist"),
                "videoId": tabs.get("video_id", ""),
                "tempo": tabs.get("tempo", 120)
            }
        })
    except Exception as e:
        logger.error(f"Error generating tabs: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500 