import requests
import json

def test_facial_analysis_api():
    """Test script for the facial analysis API"""
    
    # API endpoint
    url = "http://localhost:8000/api/facial-analysis/"
    
    # Test with a sample video file (you'll need to provide your own video file)
    video_file_path = "sample_video.mp4"  # Replace with your video file path
    
    try:
        with open(video_file_path, 'rb') as video_file:
            files = {'video': video_file}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print(f"Video file not found: {video_file_path}")
        print("Please provide a valid video file path")
    except Exception as e:
        print(f"Error testing API: {str(e)}")

if __name__ == "__main__":
    test_facial_analysis_api()