@echo off
cd /d "%~dp0"
python nightly_improve.py --accounts steak2222,steak222 --prefix steak --teacher-positions 8000 --selfplay-games 600 --late-positions 1000 --history-runs 3
