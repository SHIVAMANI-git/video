import cv2

# Your NVR details
username = "admin"
password = "CRYPTOGRAPHY@1789"
ip = "192.168.1.24"
port = 554  # Default RTSP port

# Common RTSP URL formats for CP Plus / Dahua
rtsp_urls = [
    f"rtsp://{username}:{password.replace('@', '%40')}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{username}:{password.replace('@', '%40')}@{ip}:{port}/cam/realmonitor?channel=1&subtype=1",
    f"rtsp://{username}:{password.replace('@', '%40')}@{ip}:{port}/cam/chan1",
    f"rtsp://{username}:{password.replace('@', '%40')}@{ip}:{port}/h264/ch1/main/av_stream",
    f"rtsp://{username}:{password.replace('@', '%40')}@{ip}:{port}/live/ch00_00"
]

print("🔍 Testing RTSP URLs...\n")

for url in rtsp_urls:
    print(f"Trying: {url}")
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print("✅ SUCCESS! This RTSP URL works.\n")
        cap.release()
        break
    else:
        print("❌ Failed.\n")
