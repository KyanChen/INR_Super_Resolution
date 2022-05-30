import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord, calc_psnr
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='airplane07_32.tif')
    parser.add_argument('--model', default='save/_train_UC_32_256_liif/epoch-best.pth')
    parser.add_argument('--resolution', default='128,128')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = list(map(int, args.resolution.split(',')))
    gt = transforms.ToTensor()(Image.open(args.input.replace('32', '256')).convert('RGB'))
    gt = transforms.Resize(size=(h, w))(gt)

    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
    res = calc_psnr(pred, gt)
    print(res)
