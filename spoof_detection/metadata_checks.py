"""
Metadata checks for uploaded images (EXIF, resolution, compression artifacts)
"""
from PIL import Image
from PIL.ExifTags import TAGS
import logging

logger = logging.getLogger(__name__)

def check_image_metadata(image_path):
    """
    Analyze image metadata for suspicious patterns
    Returns: dict with flags
    """
    try:
        img = Image.open(image_path)
        
        # Extract EXIF
        exif_data = img._getexif()
        exif = {}
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                exif[tag] = value
        
        # Check resolution (very high or very low can be suspicious)
        width, height = img.size
        resolution_flag = width < 200 or height < 200 or width > 4000 or height > 4000
        
        # Check for edited timestamp
        timestamp_flag = 'DateTime' in exif and 'DateTimeOriginal' in exif and exif['DateTime'] != exif['DateTimeOriginal']
        
        return {
            'resolution_suspicious': resolution_flag,
            'timestamp_mismatch': timestamp_flag,
            'exif_data': exif
        }
    except Exception as e:
        logger.error(f"Metadata check error: {e}")
        return {'error': str(e)}