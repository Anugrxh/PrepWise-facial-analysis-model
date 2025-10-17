#!/usr/bin/env python
"""
Startup script for the Facial Analysis Django API
"""
import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(e.stderr)
        return False

def main():
    print("🚀 Starting Facial Analysis API Server")
    print("=" * 50)
    
    # Check if model file exists
    model_path = "emotion_detction_model.h5"
    if not os.path.exists(model_path):
        print(f"⚠️  Warning: Model file '{model_path}' not found in project root")
        print("Please ensure your emotion detection model is in the project directory")
        print()
    
    # Run migrations
    if not run_command("python manage.py migrate", "Running database migrations"):
        print("Migration failed. Please check the error above.")
        return
    
    # Collect static files (if needed)
    run_command("python manage.py collectstatic --noinput", "Collecting static files")
    
    print("\n🎯 API Endpoint will be available at:")
    print("   POST http://localhost:8000/api/facial-analysis/")
    print("\n📝 Expected request format:")
    print("   Content-Type: multipart/form-data")
    print("   Field: 'video' (video file)")
    print("\n🔧 To test the API, use the test_api.py script")
    print("\n" + "=" * 50)
    
    # Start the development server
    print("Starting Django development server...")
    try:
        subprocess.run("python manage.py runserver", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()