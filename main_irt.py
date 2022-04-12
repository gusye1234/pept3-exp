import os
import argparse
import torch
import json
from ms import helper
import sys
from ms.model import TransPro, PrositIRT, TransProIRT
from tqdm import tqdm
from ms.dataset import FragDataset, IrtDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--fraglayer", type=int, default=1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--innerdim", type=int, default=256)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--testbatch", type=int, default=1024)
    parser.add_argument("--load", type=int, default=0)
    args = parser.parse_args()
    return args

args = parse_args()

helper.set_seed(2021)

TRAIN_EPOCHS = 300
if args.test:
    TRAIN_EPOCHS = 0
TRAIN_BATCH_SIZE = args.batch
PRED_BATCH_SIZE = args.testbatch

xlabel = ["sequence_integer",
          "precursor_charge_onehot",
          "collision_energy_aligned_normed"]
ylabel = "intensities_raw"

# Load data
config_data = json.load(open("./checkpoints/data.json"))
frag_dir = config_data['frag']
train_val = os.path.join(frag_dir, "traintest_hcd.hdf5")
holdout = os.path.join(frag_dir, "holdout_hcd.hdf5")
irt_data = config_data['irt']

# train_data = FragDataset(holdout, ratio=0.8)
train_data = IrtDataset(irt_data)

train_loader = DataLoader(train_data.train(), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(train_data.valid(), batch_size=PRED_BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = DataLoader(
    train_data.test(), batch_size=PRED_BATCH_SIZE, shuffle=True, drop_last=True)

# Prepare model stuffs

if torch.cuda.is_available():
    gpu_index = args.gpu
    device = torch.device(f"cuda:{gpu_index}")
else:
    device = torch.device("cpu")
print("Run on", device)

# model = TransProIRT().to(device)
model = PrositIRT().to(device)
print(model.comment())
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, eps=1e-8)

interval = 100
choice = 'irt'
best_loss = torch.inf
stopper = helper.EarlyStop(pat=20)
writer = SummaryWriter(logdir=f"./logs/{model.comment()}-{args.batch}")
# if args.load and os.path.exists(f"./checkpoints/best/best_valid_frag_{model.comment()}-{args.batch}.pth"):
#     model.load_state_dict(torch.load(
#         f"./checkpoints/best/best_valid_frag_{model.comment()}-{args.batch}.pth", map_location=device))
#     print("Load")
for epoch in range(TRAIN_EPOCHS):
    loss = 0.
    train_count = 0
    model = model.train()
    print(f"Epoch [{epoch+1:3d}/{TRAIN_EPOCHS}]", end=": ")
    for i, data in enumerate(train_loader):
        train_count += 1

        data = {k : v.to(device) for k, v in data.items()}
        data["peptide_mask"] = helper.create_mask(data['sequence_integer'])     
        # print({k: v.shape for k, v in data.items()})
        pred = model(data, choice='irt')
        # print(data['irt'].shape, pred.shape)
        loss_b = helper.mse_distance(data['irt'].squeeze(), pred.squeeze())
        loss += loss_b.item()
        print(f"\r    -Train Loss {loss/train_count:.5f}", end="")
        sys.stdout.flush()
        optimizer.zero_grad()
        loss_b.backward()
        optimizer.step()
    loss /= train_count

    with torch.no_grad():
        loss_test = 0
        test_count = 0
        model = model.eval()
        for i, data in enumerate(valid_loader):
            test_count += 1

            data = {k: v.to(device) for k, v in data.items()}
            data["peptide_mask"] = helper.create_mask(data['sequence_integer'])

            pred = model(data, choice='irt')
            # print(torch.concat([data['irt'][:20], pred[:20]], dim=1))
            loss_b = helper.mse_distance(data['irt'].squeeze(), pred.squeeze())
            loss_test += loss_b.item()
        loss_test /= test_count
        print(f"    -Valiad Loss: {loss_test:.5f}")           
        writer.add_scalars('loss', {'train':loss, 'test': loss_test}, epoch)
        if loss_test < best_loss:
            print("     -achieve best, saved")
            best_loss = loss_test
            torch.save(model.state_dict(), f"./checkpoints/irt/best_valid_irt_{model.comment()}-{args.batch}.pth")
        if stopper.step(loss_test):
            print("Trigger Early Stop")
            break

with torch.no_grad():
    print("Start test:")
    loss_test = 0
    test_count = 0
    print("Load best from",
          f"./checkpoints/irt/best_valid_irt_{model.comment()}-{args.batch}.pth")
    model.load_state_dict(torch.load(
        f"./checkpoints/irt/best_valid_irt_{model.comment()}-{args.batch}.pth", map_location="cuda:0"))
    model = model.eval()
    for i, data in enumerate(test_loader):
        test_count += 1

        data = {k: v.to(device) for k, v in data.items()}
        data["peptide_mask"] = helper.create_mask(data['sequence_integer'])

        pred = model(data, choice='irt')
        loss_b = helper.mse_distance(data['irt'].squeeze(), pred.squeeze())
        loss_test += loss_b.item()
    loss_test /= test_count
    print(f"----Test Loss: {loss_test:.5f}----")
