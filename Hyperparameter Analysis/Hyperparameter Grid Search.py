import os


def train(epochs=300, batch_size=1, learning_rate=0.0005, expansion=1, warmup=0.2, beta=0.5, name="test", loss="Fusion6"):

    str1 = "python train_nerv.py -e " + str(epochs) + " --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
            --outf " + name + " --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion " + str(expansion) + "  \
            --single_res --loss " + str(loss) + " --warmup " + str(warmup) + " --lr_type cosine --beta " + str(beta) + " --strides 5 2 2 2 2  --conv_type conv \
            -b " + str(batch_size) + "  --lr " + str(learning_rate) + " --norm none  --act swish"


    print(str1)

    os.system(str1)

def eval(epochs=300, batch_size=1, learning_rate=0.0005, expansion=1, warmup=0.2, beta=0.5, name="test", loss="Fusion6", dump_images=False):
    directory = "output/" + name + "/bunny"
    subfolders = next(os.walk(directory))[1]

    for f in subfolders:
        if "eval" not in f:
            file = directory + "/" + f + "/model_latest.pth"
            break

    extra = ""
    if dump_images:
        extra += " --dump_images"

    str1 = "python train_nerv.py -e " + str(epochs) + " --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
                --outf " + name + " --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion " + str(expansion) + "  \
                --single_res --loss " + str(loss) + " --warmup " + str(warmup) + " --lr_type cosine --beta " + str(beta) + " --strides 5 2 2 2 2  --conv_type conv \
                -b " + str(batch_size) + "  --lr " + str(learning_rate) + " --norm none  --act swish --weight " + file + " --eval_only " + extra



    print(str1)

    os.system(str1)


os.chdir("../")


do_train_hyperparams = True
do_eval_hyperparams = False

if do_train_hyperparams:
    beta = [0.5, 0.9]

    batch_size = [1, 3]

    warmup = [0.1, 0.2]

    loss_type = ["Fusion6", "Fusion5", "L2"]

    for be in beta:
        for w in warmup:
            for loss in loss_type:
                for ba in batch_size:
                    name = "be" + str(be) + "_w" + str(w) + "_loss" + str(loss) + "_ba" + str(ba)
                    train(name=name, learning_rate=5e-4, batch_size=ba, epochs=300, loss=loss, warmup=w, beta=be)


if do_eval_hyperparams:
    beta = [0.5, 0.9]

    batch_size = [1, 3]

    warmup = [0.1, 0.2]

    loss_type = ["Fusion6", "Fusion5", "L2"]

    for be in beta:
        for w in warmup:
            for loss in loss_type:
                for ba in batch_size:
                    name = "be" + str(be) + "_w" + str(w) + "_loss" + str(loss) + "_ba" + str(ba)
                    if name == "be0.9_w0.1_lossL2_ba1" or name == "be0.5_w0.2_lossFusion6_ba1":
                        eval(name=name, learning_rate=5e-4, batch_size=ba, epochs=300, loss=loss, warmup=w, beta=be, dump_images=True)