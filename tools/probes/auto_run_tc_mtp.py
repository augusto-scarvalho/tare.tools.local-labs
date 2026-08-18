import time
import os
import subprocess

target_file = "/home/augus/models/thinkingcap-27b-mtp/ThinkingCap-Qwen3.6-27B-Q4_K_M-MTP.gguf"
target_size = 16000000000  # ~16GB

print("Waiting for ThinkingCap MTP download to complete...")
while True:
    wsl_path = r"\\wsl.localhost\Ubuntu-24.04\home\augus\models\thinkingcap-27b-mtp\ThinkingCap-Qwen3.6-27B-Q4_K_M-MTP.gguf"
    if os.path.exists(wsl_path):
        sz = os.path.getsize(wsl_path)
        print(f"Current size: {sz / (1024*1024):.1f} MB / 16031.9 MB", end="\r")
        if sz >= 16000000000:
            print("\nDownload complete! Launching market benchmark for thinkingcap-27b-mtp-q4 with MTP...")
            break
    time.sleep(10)

cmd = [
    "powershell", "-ExecutionPolicy", "Bypass", "-File", "ops\\run_market_bench.ps1",
    "-Model", "thinkingcap-27b-mtp-q4", "-Spec", "draft-mtp", "-HumanEvalN", "60", "-Gsm8kN", "200", "-Tag", "market-r0"
]
subprocess.run(cmd, check=True)
