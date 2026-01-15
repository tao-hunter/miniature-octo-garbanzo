#!/bin/bash
set -e

echo "-----------------------------------------------------"
echo "🚀 STARTING MAIN FASTAPI SERVICE (Base Env)"
echo "-----------------------------------------------------"

# 3. Khởi chạy App chính (Foreground)
# App chính chạy trên Base Python (System Python)
exec python serve.py