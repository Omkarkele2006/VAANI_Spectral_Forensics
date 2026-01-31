import requests

# The URL of your running Flask server
url = 'http://127.0.0.1:5000/analyze'

# The file we want to upload 
file_path = 'test_audio.wav'

try:
    print(f"Sending {file_path} to VAANI Server...")
    
    # Open the file in binary mode ('rb') and send it
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    # Print the server's reply
    print("\nServer Response:")
    print(response.json())

except Exception as e:
    print(f"\nError: {e}")
    print("Is the server running in the other terminal?")