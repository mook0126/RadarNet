import os
import torch
import torch.optim as optim
from datasets.nuscenes import nuScenes
from backbone import Backbone
from header import Header
from loss import calculate_loss
from utils import NMS, output_process
from evaluate import evaluate_result

def main(opt):

    num_chan = 38
    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_path = "/home/mook/RadarNet-pytorch/train_result/"
    data_path = "/home/mook/RadarNet-pytorch/data/"
    model_path = base_path + "model/"
    loss_path = base_path + "loss.txt"

    load_checkpoint = True   # 체크포인트에서 학습 이어서 하기 가능하게 설정

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print('Creating model...')
    BBNet = Backbone(num_chan).to(device)
    Header_car = Header().to(device)

    optimizer_b = optim.Adam(BBNet.parameters(), lr=1e-4)  # 학습률 수정 (1e-2 → 1e-4)
    optimizer_hc = optim.Adam(Header_car.parameters(), lr=1e-4)

    scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=5, gamma=0.8)
    scheduler_hc = torch.optim.lr_scheduler.StepLR(optimizer_hc, step_size=5, gamma=0.8)

    st_epoch = 1  # 시작 epoch

    # 🚀 체크포인트 불러오기 (가장 최신 모델 이어서 학습)
    if load_checkpoint:
        if os.path.exists(model_path):
            model_files = sorted(os.listdir(model_path), key=lambda x: int(x.split("-")[1].split(".")[0]))
            if model_files:
                latest_checkpoint = model_files[-1]
                print('Loading checkpoint model:', latest_checkpoint)

                checkpoint = torch.load(model_path + latest_checkpoint, map_location=device)
                BBNet.load_state_dict(checkpoint["backbone"])
                optimizer_b.load_state_dict(checkpoint["optimizer_b"])
                Header_car.load_state_dict(checkpoint["header"])
                optimizer_hc.load_state_dict(checkpoint["optimizer_hc"])
                st_epoch = checkpoint["epoch"] + 1  # 저장된 epoch 다음부터 시작
            else:
                print("No checkpoint found, starting from scratch...")
        else:
            print("No checkpoint directory found, starting from scratch...")

    print('Setting up train & validation data...')
    train_loader = torch.utils.data.DataLoader(nuScenes(opt, opt.train_split, data_path), batch_size=1, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(nuScenes(opt, opt.train_split, data_path), batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    print('Batch_size is:', batch_size)

    for epoch in range(st_epoch, opt.num_epochs + 1):
        print(f'Starting training for epoch-{epoch}')

        BBNet.train()
        Header_car.train()
        loss_epoch = torch.zeros(1).to(device)
        ap_epoch = torch.zeros(1).to(device)

        for ind, (gt, voxel, _) in enumerate(train_loader):
            if gt.size()[1] == 0:
                continue

            voxel = voxel.float().to(device)
            backbones = BBNet(voxel)
            cls_cars, reg_cars = Header_car(backbones)

            car_dets = output_process(cls_cars, reg_cars, device, batch_size)
            gt_car = gt.to(device)
            match_label_car = matching_boxes(car_dets[:, 6:8], gt_car[:, 0:5], device)
            loss_car = calculate_loss(match_label_car, car_dets, gt_car, device)

            optimizer_b.zero_grad()
            optimizer_hc.zero_grad()
            loss_car.backward()
            optimizer_b.step()
            optimizer_hc.step()

            loss_epoch += loss_car.item()
            with torch.no_grad():
                car_output = NMS(car_dets, 0.05, 200)
                AP = evaluate_result(car_output, gt_car, device)
                ap_epoch += AP.item()

            print(f'The loss of iter-{ind} is {loss_car.item()}')
            print(f'The AP of iter-{ind} is {AP.item()}')

        loss_epoch /= len(train_loader)
        ap_epoch /= len(train_loader)
        print(f'The loss of epoch-{epoch} is {loss_epoch}')
        print(f'The AP of epoch-{epoch} is {ap_epoch}')

        # ✅ 중간마다 weight 저장 (체크포인트 저장)
        state = {
            'backbone': BBNet.state_dict(),
            'optimizer_b': optimizer_b.state_dict(),
            'header': Header_car.state_dict(),
            'optimizer_hc': optimizer_hc.state_dict(),
            'epoch': epoch
        }
        torch.save(state, model_path + f'epoch-{epoch}.pth')

        scheduler_b.step()
        scheduler_hc.step()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
