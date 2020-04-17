import os

for f in os.listdir("./"):
    if f.endswith(".txt"):
        open(f, 'w').close()
