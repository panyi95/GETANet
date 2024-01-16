import os
import torch
from torch.autograd import Variable
from model import mMMCCN_IAWT
from PIL import Image
import h5py
import cv2
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count

def eval_game(output, target, L=0):
    assert output.shape == target.shape
    H, W = target.shape

    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

            abs_error += abs(output_block.sum()-target_block.sum().float())
            square_error += (output_block.sum()-target_block.sum().float()).pow(2)

    return abs_error, square_error


rgb_mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
thermal_mean_std = ([0.5, 0.5, 0.5], [0.18362635145048645, 0.18362635145048645, 0.18362635145048645])



rgb_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*rgb_mean_std)
])
thermal_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*thermal_mean_std)
])

model_path = 'data/final.pth'
save_dir = 'result'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, 'txt_files'))
    os.mkdir(os.path.join(save_dir, 'vis'))
    os.mkdir(os.path.join(save_dir, 'den'))


def test(file_list, model_path):
    device = torch.device(0)
    net = mMMCCN_IAWT().to(device)
    state_dict = torch.load(model_path)
    new_dict = OrderedDict()
    for key, value in state_dict.items():
        new_dict[key[4:]] = value
    net.load_state_dict(new_dict)
    net.eval()

    maes = AverageMeter()
    mses = AverageMeter()
    game_mae = [0, 0, 0, 0]
    game_mse = [0, 0, 0, 0]
    name_list = []
    pred_list = []
    gt_list = []
    mae_list =[]
    mse_list = []
    game0_list  = []
    game1_list  = []
    game2_list  = []
    game3_list  = []



    for idx, filename in enumerate(file_list):
        # read and process data
        fname = filename.split('.')[0]
        print(idx, fname)
        rgb_path = os.path.join('data', 'rgb', fname + '.jpg' )
        rgb_img = Image.open(rgb_path)
        thermal_path = os.path.join('data', 'thermal', fname + 'R.jpg' )
        thermal_img = Image.open(thermal_path)
        if thermal_img.mode == 'L':
            thermal_img = thermal_img.convert('RGB')
        den = h5py.File(os.path.join('data', 'dm_5', fname + '.h5'))
        den = np.asarray(den['density'])
        den = den.astype(np.float32)
        thermal_img = thermal_img_transform(thermal_img)
        rgb_img = rgb_img_transform(rgb_img)

        with torch.no_grad():
            thermal_img = Variable(thermal_img[None, :, :, :]).to(device)
            rgb_img = Variable(rgb_img[None,:,:,:]).to(device)

            pred_map = net([rgb_img,thermal_img])
            pred_map_tensor = pred_map[0][0].cpu() / 100.

            pred_map = pred_map.data.cpu().numpy()
            save_map = pred_map[0][0] / 100.

            pred_cnt = np.sum(pred_map) / 100.
            gt_count = np.sum(den)

            mae = abs(gt_count - pred_cnt)
            mse = (gt_count - pred_cnt) * (gt_count - pred_cnt)

            single_game_mae = []
            single_game_mse = []

            for L in range(4):
                abs_error, square_error = eval_game(pred_map_tensor, torch.from_numpy(den), L)
                game_mae[L] += abs_error.item()
                game_mse[L] += square_error.item()
                single_game_mae.append(abs_error.item())
                single_game_mse.append(square_error.item())

            # vis blend
            ori_img = cv2.imread(thermal_path)
            heatmap = save_map
            heatmap = heatmap / np.max(heatmap)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            x, y = heatmap.shape[0:2]
            ori_img = cv2.resize(ori_img, (y, x))
            heatmap = 0.9 * heatmap + ori_img
            cv2.imwrite(os.path.join(save_dir, 'vis', '{}_{}_{}.png'.format(fname, mae, mse)), heatmap)

            # vis density
            den_map =  save_map / np.max(save_map)
            den_map = np.uint8(255*den_map)
            den_map = cv2.applyColorMap(den_map, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, 'den', '{}_{}_{}.png'.format(fname, mae, mse)), den_map)


            print('pred: %.4f, gt: %.4f, mae: %.4f, mse: %.4f' % (pred_cnt, gt_count, mae, mse))
            print('gname0: %.4f, gname1: %.4f, gname2: %.4f, gname3: %.4f' % (single_game_mae[0], single_game_mae[1], single_game_mae[2], single_game_mae[3]))
            name_list.append(fname)
            pred_list.append(pred_cnt)
            gt_list.append(gt_count)
            mae_list.append(mae)
            mse_list.append(mse)
            game0_list.append(single_game_mae[0])
            game1_list.append(single_game_mae[1])
            game2_list.append(single_game_mae[2])
            game3_list.append(single_game_mae[3])
            maes.update(mae)
            mses.update(mse)

    name_list = [int(name) for name in name_list]
    name_array = np.array(name_list)
    order = np.argsort(name_array)
    name_array = name_array[order]
    mse_array = np.array(mse_list)[order]
    mae_array = np.array(mae_list)[order]
    gt_array = np.array(gt_list)[order]
    pred_array = np.array(pred_list)[order]
    game0_array = np.array(game0_list)[order]
    game1_array = np.array(game1_list)[order]
    game2_array = np.array(game2_list)[order]
    game3_array = np.array(game3_list)[order]

    final_mae = maes.avg
    final_mse = np.sqrt(mses.avg)
    N = len(file_list)
    game_mae = [m / N for m in game_mae]
    game_mse = [np.sqrt(np.array(m / N)) for m in game_mse]
    print('GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f} '
          .format(game0=game_mae[0], game1=game_mae[1], game2=game_mae[2],
                  game3=game_mae[3], mse=game_mse[0]))


    f = open(os.path.join(save_dir, 'txt_files', 'test_results.txt'), 'w')
    for fname, gt, pred, mae, mse, game0, game1, game2, game3 in zip(name_array, gt_array, pred_array, mae_array,
                                                                         mse_array, game0_array, game1_array,
                                                                         game2_array, game3_array):
        f.write(str(fname) + ',%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f'%(gt, pred, mae, mse, game0, game1, game2, game3) + '\n')
    f.write("final results: " + 'MAE: %.2f, MSE: %.2f'%(final_mae, final_mse) + '\n')
    f.write('GAME_MAE: %.2f, %.2f, %.2f, %.2f'%(game_mae[0], game_mae[1], game_mae[2],game_mae[3])+ '\n')
    f.write('GAME_MSE: %.2f, %.2f, %.2f, %.2f'%(game_mse[0], game_mse[1], game_mse[2],game_mse[3]))
    print(final_mae, final_mse)
    print('game_mae:', game_mae)
    print('game_mse:', game_mse)

    f.close()




if __name__ == '__main__':
    file_list = [filename for root, dirs, filename in os.walk('data' + '/dm_5')]
    test(file_list[0], model_path)
 
