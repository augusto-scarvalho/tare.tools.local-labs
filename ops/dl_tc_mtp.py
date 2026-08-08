import urllib.request
import os

url = "https://huggingface.co/protoLabsAI/ThinkingCap-Qwen3.6-27B-MTP-GGUF/resolve/main/ThinkingCap-Qwen3.6-27B-Q4_K_M-MTP.gguf"
dest_dir = "/home/augus/models/thinkingcap-27b-mtp"
dest_file = os.path.join(dest_dir, "ThinkingCap-Qwen3.6-27B-Q4_K_M-MTP.gguf")

os.makedirs(dest_dir, exist_ok=True)
existing_size = os.path.getsize(dest_file) if os.path.exists(dest_file) else 0

req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={existing_size}-"})
print(f"Resuming download from byte {existing_size}...")

try:
    with urllib.request.urlopen(req) as resp, open(dest_file, "ab") as f:
        total = int(resp.headers.get("Content-Length", 0)) + existing_size
        downloaded = existing_size
        chunk_size = 1024 * 1024 * 8  # 8MB chunks
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            pct = (downloaded / total) * 100 if total else 0
            print(f"Downloaded {mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%)", end="\r")
    print("\nResume complete!")
except Exception as e:
    print(f"\nError: {e}")
