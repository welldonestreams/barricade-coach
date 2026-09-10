@echo off
cd /d "%~dp0"
python nightly_improve.py --accounts steak2222 --prefix steak --teacher-positions 5000 --selfplay-games 300 --late-positions 500
