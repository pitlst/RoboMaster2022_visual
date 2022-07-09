rem windows一般为调试用，程序不需要循环挂起
call conda activate cuda
cd C:\Program Files (x86)\Intel\openvino_2022.1.0.643
c:
call setupvars.bat
d:
python main_energy.py
pause