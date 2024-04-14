import os



def train(epochs=1200, batch_size=1, learning_rate=0.0005, expansion=1, warmup=0.1, beta=0.9, name="test", loss="L2", dataset="colon"):

    str1 = "python train_nerv.py -e " + str(epochs) + " --lower-width 96 --num-blocks 1 --dataset " + dataset + " --frame_gap 1 \
            --outf " + name + " --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion " + str(expansion) + "  \
            --single_res --loss " + str(loss) + " --warmup " + str(warmup) + " --lr_type cosine --beta " + str(beta) + " --strides 5 3 2 2 1  --conv_type conv \
            -b " + str(batch_size) + "  --lr " + str(learning_rate) + " --norm none  --act swish"


    print(str1)

    os.system(str1)


def eval(epochs=300, batch_size=1, learning_rate=0.0005, expansion=1, warmup=0.1, beta=0.9, name="test", loss="L2", dataset="colon", dump_images=False, pruned=True):

    extra = ""

    if pruned == True:
        name = name + "Pruned"
        extra += "--quant_bit 8 --quant_axis 0 --suffix 107"

    if dump_images:
        extra += " --dump_images"

    directory = "output/" + name + "/bunny"
    subfolders = next(os.walk(directory))[1]

    for f in subfolders:
        if "eval" not in f:
            file = directory + "/" + f + "/model_latest.pth"
            break

    str1 = "python train_nerv.py -e " + str(epochs) + " --lower-width 96 --num-blocks 1 --dataset " + dataset + " --frame_gap 1 \
            --outf " + name + " --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion " + str(expansion) + "  \
            --single_res --loss " + str(loss) + " --warmup " + str(warmup) + " --lr_type cosine --beta " + str(beta) + " --strides 5 3 2 2 1  --conv_type conv \
            -b " + str(batch_size) + "  --lr " + str(learning_rate) + " --norm none  --act swish --weight " + file + " --eval_only "+ extra

    print(str1)

    os.system(str1)

def prune(epochs=50, batch_size=1, learning_rate=0.0005, expansion=1, warmup=0.1, beta=0.9, name="test", loss="L2", dataset="colon"):
    directory = "output/" + name + "/" + dataset
    first_folder = [x[0] for x in os.walk(directory)]
    directory = first_folder[1]

    str1 = "python train_nerv.py -e " + str(epochs) + " --lower-width 96 --num-blocks 1 --dataset " + dataset + " --frame_gap 1 \
            --outf " + name + "Pruned --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion " + str(expansion) + "  \
            --single_res --loss " + str(loss) + " --warmup " + str(warmup) + " --lr_type cosine --beta " + str(beta) + " --strides 5 3 2 2 1  --conv_type conv \
            -b " + str(batch_size) + "  --lr " + str(learning_rate) + " --norm none  --act swish --weight " + directory + "/model_latest.pth --not_resume_epoch --prune_ratio 0.4 "


    print(str1)

    os.system(str1)

os.chdir("../")

train(epochs=1200)

#prune()


#eval(epochs=1200, dump_images=True, pruned=True)