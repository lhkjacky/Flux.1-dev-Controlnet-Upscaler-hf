call C:\ProgramData\anaconda3\Scripts\activate.bat
call activate flux_upscaler
cd /d D:\Flux.1-dev-Controlnet-Upscaler-hf\
pip install protobuf
python app_cpu.py
cmd.exe