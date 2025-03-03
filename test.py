import requests

# Fetch the ngrok status page
try:
    response = requests.get("http://127.0.0.1:4040/api/tunnels")
    if response.status_code == 200:
        data = response.json()
        public_url = data["tunnels"][0]["public_url"]
        print("Public URL: ", public_url)
    else:
        print("Unable to fetch the URL, ngrok may not be running.")
except Exception as e:
    print(f"Error fetching ngrok public URL: {e}")