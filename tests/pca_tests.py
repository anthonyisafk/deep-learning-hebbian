import subprocess

if __name__ == '__main__':
    exec = "C:/Users/tonyt/anaconda3/python.exe .\src\smoking_pca.py"
    for n in range(1, 16):
        subprocess.call(exec + f" -nc {n}")