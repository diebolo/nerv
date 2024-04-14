import matplotlib.pyplot as plt
import os
from py_markdown_table.markdown_table import markdown_table

os.chdir("../")

beta = [0.5, 0.9]

batch_size = [1, 3]

warmup = [0.1, 0.2]

loss_type = ["Fusion6", "Fusion5", "L2"]
bestest_psnr = 0
seto = False

table = []

for be in beta:
    for w in warmup:
        for loss in loss_type:
            for ba in batch_size:
                name = "be" + str(be) + "_w" + str(w) + "_loss" + str(loss) + "_ba" + str(ba)
                print(name)
                directory = "output/" + name + "/bunny"
                subfolders = next(os.walk(directory))[1]

                for f in subfolders:
                    if "eval" not in f:
                        file = directory + "/" + f + "/rank0.txt"
                        break

                psnr = []
                epoch = []
                mssim = []

                curr_epoch = 0

                with open(file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        # check if it contains 132/132
                        if "132/132" in line or "44/44" in line:
                            curr_epoch += 1
                            substring = line[line.index('PSNR') + 5:]
                            number = float(substring[:substring.index(',')])
                            psnr.append(number)
                            epoch.append(curr_epoch)


                            substring2 = line[line.index('MSSSIM') + 7:]
                            number2 = float(substring2[:substring2.index('\n')])
                            mssim.append(number2)



                table.append({
                                    "Beta": be,
                                    "Warmup": w,
                                    "Loss": loss,
                                    "Batch": ba,
                                    "PSNR": psnr[-1],
                                    "MSSIM": mssim[-1]
                                })


                if name == "be0.5_w0.2_lossFusion6_ba1":
                    og_psnr = psnr.copy()
                    og_epoch = epoch.copy()
                else:
                    if psnr[-1] > bestest_psnr:
                        bestest_psnr = psnr[-1]
                        best_psnr = psnr.copy()
                        best_epoch = epoch.copy()
                        print(name)

                    if ba == 3:
                        if seto:
                            plt.plot(epoch, psnr, color="red")
                        else:
                            seto = True
                            plt.plot(epoch, psnr, color="red", label="Batch size = 3")
                    else:
                        plt.plot(epoch, psnr, color="C0")


plt.legend()
plt.plot(og_epoch, og_psnr, color="orange", linewidth=3, label="Default")
plt.plot(best_epoch, best_psnr, color="green", linewidth=3, label="Best")
#invert y axis
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.show()


print("\n\n\n\n\n")
markdown = markdown_table(table).get_markdown()
print(markdown)