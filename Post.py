from shutil import copyfile


src = "./cmake-build-release/"
dst = "./Python/"

files = [
    "Ising.py",
    "_Ising.so",
]

for file in files:
    copyfile(src + file, dst + file)
