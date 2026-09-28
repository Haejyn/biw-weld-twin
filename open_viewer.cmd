@echo off
rem 디지털 트윈 화면 열기 — viewer 폴더를 8790 포트로 띄우고 브라우저를 연다
cd /d "%~dp0viewer"
start "" http://localhost:8790/
"%~dp0.venv\Scripts\python.exe" -m http.server 8790
