import subprocess

if __name__ == '__main__':
    exec = "C:/Users/tonyt/anaconda3/python.exe .\src\smoking.py"
    for eta in [0.05, 0.10]:
        for n in range(1, 11):
            subprocess.call(exec + f" -eta {eta} -nc {n}")