import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_checkpoint, save_images, save_checkpoint, DDPMDataset, MaskDataset, DDPMDataset_sep, DDPMDataset_csv_generate
from DDPM_model import DDPM, Discriminator

torch.manual_seed(9)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.args = args

        self.beta = self.prepare_linear_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_linear_noise_schedule(self):
        return torch.abs(torch.cos(torch.linspace(0, torch.pi/2, self.noise_steps))*self.beta_end -
                         (self.beta_end-self.beta_start))
        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, labels, t):
        labels = labels[:labels.shape[0]]

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        eps = torch.randn_like(labels)
        img = sqrt_alpha_hat * labels + sqrt_one_minus_alpha_hat * eps
        return img, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def pre_output(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample(self, model, labels, cls, disc=None, epoch=None, n=None):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.args.img_channel, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, mask=labels, cls=cls)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
                # xs = ((x.clamp(-1, 1) + 1) / 2)
                # xs[xs >= 0.25] = torch.ones_like(xs[xs >= 0.25])
                # xs[xs < 0.25] = torch.zeros_like(xs[xs < 0.25])
                # xs = (xs * 255).type(torch.uint8)
                # save_images(xs, os.path.join("results/steps", f"{i}_fake.png"))
        model.train()

        # score = disc(torch.clone(x.clamp(-1, 1)))
        # label, vessel, roi, od = x[:,0,...], x[:,1,...], x[:,2,...], x[:,3,...]
        MA, EX, HE, SE = x[:, 0, ...].unsqueeze(1), x[:, 1, ...].unsqueeze(1), x[:, 2, ...].unsqueeze(1), x[:, 3, ...].unsqueeze(1)
        # VE = x[:, 4, ...].unsqueeze(1)
        # x[x >= 0.25] = torch.ones_like(x[x >= 0.25])
        # x[x < 0.25] = torch.zeros_like(x[x < 0.25])

        # label, vessel, roi, od = self.pre_output(label), self.pre_output(label), self.pre_output(label), self.pre_output(label)
        MA, EX, HE, SE = self.pre_output(MA), self.pre_output(EX), self.pre_output(HE), self.pre_output(SE)
        # VE = self.pre_output(VE)
        labels = self.pre_output(labels)
        if not os.path.exists(os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}")):
            os.makedirs(os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}"))
        save_images(labels, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{counter}_label.png"))
        save_images(MA, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{counter}_MA.png"))
        save_images(EX, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{counter}_EX.png"))
        save_images(HE, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{counter}_HE.png"))
        save_images(SE, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{counter}_SE.png"))
        # save_images(VE, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{counter}_VE.png"))

        # return score

    def generate(self, model, labels=None, cls=None, n=0, counter=0,
                 name=None, idx=None, iteration=0):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.args.img_channel, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, mask=labels, cls=cls)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
        model.train()
        for i in range(n):
            MA, EX, HE, SE = (x[i, 0, ...][None].unsqueeze(1), x[i, 1, ...][None].unsqueeze(1),
                              x[i, 2, ...][None].unsqueeze(1), x[i, 3, ...][None].unsqueeze(1))
            labels_out = self.pre_output(labels[i, ...])

            MA, EX, HE, SE = self.pre_output(MA), self.pre_output(EX), self.pre_output(HE), self.pre_output(SE)
            if not os.path.exists(os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}")):
                os.makedirs(os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}"))
            save_images(labels_out, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{int(idx[i])}-{iteration}-{name[i]}_label.png"))
            save_images(MA, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{int(idx[i])}-{iteration}-{name[i]}_MA.png"))
            save_images(EX, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{int(idx[i])}-{iteration}-{name[i]}_EX.png"))
            save_images(HE, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{int(idx[i])}-{iteration}-{name[i]}_HE.png"))
            save_images(SE, os.path.join("/data2/xiaoyi/DR_lesions/result", f"{self.args.save_path}/{int(idx[i])}-{iteration}-{name[i]}_SE.png"))

        return counter1, counter2


def test(args):
    global counter1, counter2
    device = args.device
    label_paths = "/data2/xiaoyi/DR_lesions/FGADR/Healthy/mask_1024_jpg_structure"
    cls_paths = "/data2/xiaoyi/DR_lesions/FGADR/Healthy/Generation_lesion_3.csv"

    dataset = DDPMDataset_csv_generate(label_paths=label_paths, cls_path=cls_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pbar = tqdm(dataloader)
    ddpm = DDPM(img_channels=args.input_channel, time_dim=args.emb_dim, out_channel=args.img_channel, device=args.device, num_classes=5).to(device)  # channels = 1
    diffusion = Diffusion(noise_steps=args.noise_step, img_size=args.image_size, device=device,args=args)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoints, "ddpm408.pth.tar"), ddpm, None, None, device=device
        )

    for i in range(test_args.num_iters):
        iteration = i
        for i, (labels,cls, name, idx) in enumerate(pbar):
            structures = labels.to(device)
            cls = cls.to(device)
            diffusion.generate(ddpm, labels=structures, cls=cls, n=args.batch_size, name=name, idx=idx, iteration=iteration)

def train(args):
    global counter
    device = args.device
    MA_paths = glob("/data2/xiaoyi/DR_lesions/FGADR/All/mask_1024_jpg_MA/*.jpg")
    label_paths = glob("/data2/xiaoyi/DR_lesions/FGADR/All/mask_1024_jpg_structure/*.jpg")
    cls_paths = "/data2/xiaoyi/DR_lesions/FGADR/All/DR_Seg_Grading_Label.csv"
    # dataset = DDPMDataset(data_paths=data_paths, label_paths=label_paths, img_size=args.image_size)
    dataset = DDPMDataset_sep(data_paths=MA_paths, label_paths=label_paths, cls_path=cls_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    ddpm = DDPM(img_channels=args.input_channel, time_dim=args.emb_dim, out_channel=args.img_channel, device=args.device, num_classes=5).to(device)  # channels = 1
    optimizer = optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoints, "ddpm294.pth.tar"), ddpm, optimizer, args.lr, device=args.device
        )

    disc = Discriminator(in_channels=1).to(device)
    # load_checkpoint(os.path.join(args.checkpoints, "disc6.pth.tar"), disc, None, None)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    bce = nn.BCELoss()

    m = nn.Sigmoid()
    min_avg_loss = float("inf")

    diffusion = Diffusion(noise_steps=args.noise_step, img_size=args.image_size, device=device,args=args)
    score = torch.tensor([0]).item()
    for epoch in range(1, args.epochs):
        pbar = tqdm(dataloader)
        avg_loss = 0
        count1 = 0
        count2 = 0
        for i, (images, labels, cls) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            cls = cls.to(device)
            # labels[labels > 0] = images[labels > 0]

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            for rep in range(5):
                if rep == 1:
                    count1 += 1
                predicted_noise = ddpm(x=x_t, t=t, mask=labels, cls=cls)
                loss = mse(noise, predicted_noise) + l1(noise, predicted_noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if loss < min_avg_loss:
                    if rep == 4:
                        count2 += 1
                    break

            avg_loss += loss.item()
            # kl_div_loss = kl_div.item()
            pbar.set_postfix(epoch=epoch, AVG_LOSS=avg_loss / (i + 1), count1=count1,
                             count2=count2, MIN_LOSS=min_avg_loss, classifier_score=score)

            if i % ((len(dataloader)-1)//2) == 0 and i != 0:
            #utaif i >1:

                diffusion.sample(model=ddpm, labels=labels, cls=cls, n=labels.shape[0])  # score =
                # score = torch.mean(score).item()
                counter += 1

        if min_avg_loss > avg_loss / len(dataloader):
            min_avg_loss = avg_loss / len(dataloader)
            if not os.path.exists(args.checkpoints):
                os.makedirs(args.checkpoints)
            save_checkpoint(ddpm, optimizer, filename=os.path.join(args.checkpoints, f"ddpm{epoch}.pth.tar"))


if __name__ == '__main__':
    training = False
    counter = 1

    if training:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.load_model = True
        args.noise_step = 1000
        args.epochs = 500
        args.batch_size = 2
        args.emb_dim = 256 * 1
        args.image_size = 256 * 1
        args.img_channel = 4
        args.num_workers = 4
        args.input_channel = 5
        args.save_path = "experiment_9_structure&class_2_4lesion"
        args.checkpoints = "/data2/xiaoyi/DR_lesions/results/experiment_9_structure&class_2_4lesion/checkpoints"
        args.dataset_path = None
        args.generated_path = None
        args.device = "cuda:1"
        args.lr = 3e-4
        train(args)
    else:
        counter1 = 20943
        counter2 = 1
        test_parser = argparse.ArgumentParser()
        test_args = test_parser.parse_args()
        test_args.load_model = True
        test_args.noise_step = 1000

        test_args.num_iters = 2
        test_args.batch_size = 10
        test_args.emb_dim = 256 * 1
        test_args.image_size = 256 * 1
        test_args.num_workers = 4
        test_args.input_channel = 5
        test_args.img_channel = 4

        test_args.checkpoints = "/data2/xiaoyi/DR_lesions/results/experiment_9_structure&class_2_4lesion/checkpoints"
        test_args.save_path = "generation_DR_3_mask"
        test_args.device = "cuda:4"
        test(test_args)
