import matplotlib.pyplot as plt
import os

def psnr_curve(run_id=0, learning_rate=0.0005):
    directory = "output/run" + str(run_id) + "_lr" + str(learning_rate) + "/bunny"
    subfolders = next(os.walk(directory))[1]

    for f in subfolders:
        if "eval" not in f:
            file = directory + "/" + f + "/rank0.txt"
            break

    psnr = []
    epoch = []

    curr_epoch = 0
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            #check if it contains 132/132
            if "132/132" in line:
                curr_epoch += 1
                substring = line[line.index('PSNR')+5:]
                number = float(substring[:substring.index(',')])
                psnr.append(number)
                epoch.append(curr_epoch)
    plt.plot(epoch, psnr, label=learning_rate)
    #invert y axis


os.chdir("../")

lr_lst = [5e-5,3e-5,7e-5,5e-4,5e-6,5e-3]

for idx, lr in enumerate(lr_lst):
    psnr_curve(learning_rate=lr, run_id=idx)

plt.title("PSNR for different Learning rate")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.show()