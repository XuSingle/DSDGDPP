from model import Generator, Discriminator
import argparse
import torch
import torch.nn as nn
from utils import evaluate_samples, sample_ring, sample_grid
from tqdm import tqdm
import os
from torch import distributions as dis
from GDPPLoss import compute_gdpp
from dsd import DSDTraining

def run_gan(model_num, params):
    discriminator = Discriminator().to(DEVICE)

    dsd_netD = DSDTraining(discriminator, sparsity=args.sparse_D, model_type='D')
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr=params['disc_learning_rate'],
                                 betas=(params['beta1'], 0.999), eps=params['epsilon'])
    generator = Generator().to(DEVICE)
    dsd_netG = DSDTraining(generator, sparsity=args.sparse_G, model_type='G')
    gen_optim = torch.optim.Adam(generator.parameters(), lr=params['gen_learning_rate'],
                                 betas=(params['beta1'], 0.999), eps=params['epsilon'])

    criterion = nn.BCELoss()
    flag_G = 0
    pretrain = 5000

    ############################
    # Start training
    ###########################

    for i in tqdm(range(args.iter)):

        if not args.model_type == 'OG':
            if i < pretrain:
                dsd_netG.train_on_sparse = False
                dsd_netD.train_on_sparse = False
            elif i >= pretrain and flag_G < args.G_gap:
                dsd_netG.train_on_sparse = True
                dsd_netG.update_masks()
                dsd_netD.train_on_sparse = True
                dsd_netD.update_masks()
                for para in gen_optim.param_groups:
                    para['lr'] = params['gen_learning_rate']
                for para in dis_optim.param_groups:
                    para['lr'] = params['disc_learning_rate']
                flag_G = flag_G + 1
            elif i >= pretrain and flag_G < 2 * args.G_gap:
                dsd_netG.train_on_sparse = False
                dsd_netD.train_on_sparse = False
                for para in gen_optim.param_groups:
                    para['lr'] = params['gen_learning_rate'] * 0.5
                for para in dis_optim.param_groups:
                    para['lr'] = params['disc_learning_rate'] * 0.5
                flag_G = flag_G + 1
                if flag_G == 2 * args.G_gap:
                    flag_G = 0

            if i % args.G_gap == 0 and dsd_netG.train_on_sparse:
                print('Iter%d, G_Sparse' % i)
                dsd_netG.update_masks()
            elif i % args.G_gap == 0 and not dsd_netG.train_on_sparse:
                print('Iter%d, G_Dense' % i)

        noise = dis.normal.Normal(torch.zeros(params['z_dim']), torch.ones(params['z_dim'])).sample(
            sample_shape=torch.tensor([params['batch_size']])).to(DEVICE)

        if model_num % 2 == 0:
            data = sample_ring(params['batch_size']).to(DEVICE)
        else:
            data = sample_grid(params['batch_size']).to(DEVICE)

        samples = dsd_netG(noise)

        ############################
        # Update Discriminator network
        ###########################

        dsd_netD.zero_grad()

        real_score, real_h = dsd_netD(data)
        fake_score, fake_h = dsd_netD(samples.detach())

        dis_loss = torch.mean(criterion(real_score, torch.ones_like(real_score)) + criterion(
            fake_score, torch.zeros_like(fake_score)))
        dis_loss.backward()
        dis_optim.step()

        ############################
        # Update Generator network
        ###########################

        generator.zero_grad()
        fake_score, fake_h = dsd_netD(samples)
        og_gen_loss = criterion(fake_score, torch.ones_like(fake_score))
        gdpp_loss = compute_gdpp(fake_h.to('cpu'), real_h.to('cpu')).item()
        #         print(gdpp_loss)
        gen_loss = torch.mean(gdpp_loss + og_gen_loss)
        gen_loss.backward()
        gen_optim.step()

        if i % 5000 == 0:
            noise = dis.normal.Normal(torch.zeros(params['z_dim']), torch.ones(params['z_dim'])).sample(
                sample_shape=torch.tensor([params['number_evaluation_samples']])).to(DEVICE)
            samples = dsd_netG(noise).to('cpu').detach().numpy()
            data = data.to('cpu').detach().numpy()
            evaluate_samples(samples, data, model_num, i, is_ring_distribution=(model_num % 2 == 0))

    noise = dis.normal.Normal(torch.zeros(params['z_dim']), torch.ones(params['z_dim'])).sample(
        sample_shape=torch.tensor([params['number_evaluation_samples']])).to(DEVICE)
    samples = dsd_netG(noise).to('cpu').detach().numpy()
    data = data.to('cpu').detach().numpy()
    modes, high_ratio = evaluate_samples(samples, data, model_num, i + 1, is_ring_distribution=(model_num % 2 == 0))

    with open('./%d_%s_G_gap_%d_G_sparse_%.1f_D_sparse_%.1f.txt' %
              (args.model, args.model_type, args.G_gap, args.sparse_G, args.sparse_D), 'a') as f:
        f.write("Modes:" + str(modes) + '\n')
        f.write("High_ratio:" + str(high_ratio) + '\n')

if __name__ == '__main__':
    # Change the write Dir to where you want to save the models & Plots
    write_dir = './'
    if not os.path.exists(write_dir + 'models'):
        os.makedirs(write_dir + 'models')
    if not os.path.exists(write_dir + 'Plots'):
        os.makedirs(write_dir + 'Plots')

    params = dict(
        x_dim=2,
        z_dim=256,
        beta1=0.5,
        epsilon=1e-8,
        viz_every=500,
        batch_size=512,
        gen_learning_rate=1e-3,
        disc_learning_rate=1e-4,
        number_evaluation_samples=2500,
    )

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=int, default=0, required=True)
    argparser.add_argument('--iter', type=int, default=25000)
    argparser.add_argument('--G_gap', type=int, default=5)
    argparser.add_argument('--D_gap', type=int, default=5)
    argparser.add_argument('--sparse_G', type=float, default=0.3)
    argparser.add_argument('--sparse_D', type=float, default=0.3)
    argparser.add_argument('--model_type', type=str, required=True)
    args = argparser.parse_args()

    run_gan(args.model, params)


