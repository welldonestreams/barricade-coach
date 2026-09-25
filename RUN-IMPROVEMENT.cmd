@echo off
cd /d "%~dp0"
python improve.py --rounds 3 --positions 2000 --teacher-depth 3 --teacher-seconds 8 --arena-pairs 100 --arena-seconds 1 --league-games 200 --league-seconds 0.5
pause
