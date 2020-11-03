import world
import utils
from world import cprint
import torch
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import register
from register import dataset
import numpy as np

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
best_result = {'recall': np.array([0.0]),
               'precision': np.array([0.0]),
               'ndcg': np.array([0.0]),
               'auc': np.array([0.0])}
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
print(Recmodel)

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, w=w)
        print(f'{output_information}')
        print(f"Total time:{time.time() - start}")
        if (epoch + 1) % 20 == 0:
            cprint("TEST")
            tmp = Procedure.test(dataset, Recmodel, epoch, w, world.config['multicore'], best_result)
            if tmp['recall'][0] > best_result['recall'][0]:
                best_result = tmp
                torch.save(Recmodel.state_dict(), weight_file)
    print(best_result)
finally:
    if world.tensorboard:
        w.close()
