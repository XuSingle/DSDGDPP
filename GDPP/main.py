from model import Generator, Discriminator
import argparse
import torch
import torch.nn as nn
from utils import evaluate_samples, sample_ring, sample_grid
from tqdm import tqdm
import os
from torch import distributions as dis
from GDPPLoss import compute_gdpp

def run_gan(model_num):
    discriminator = Discriminator().to(DEVICE)
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr=params['disc_learning_rate'],
                                 betas=(params['beta1'], 0.999), eps=params['epsilon'])

    generator = Generator().to(DEVICE)
    gen_optim = torch.optim.Adam(generator.parameters(), lr=params['gen_learning_rate'],
                                 betas=(params['beta1'], 0.999), eps=params['epsilon'])

    criterion = nn.BCELoss()


    ############################
    # Start training
    ###########################

    for i in tqdm(range(args.iter)):
        noise = dis.normal.Normal(torch.zeros(params['z_dim']), torch.ones(params['z_dim'])).sample(
            sample_shape=torch.tensor([params['batch_size']])).to(DEVICE)

        if model_num % 2 == 0:
            data = sample_ring(params['batch_size']).to(DEVICE)
        else:
            data = sample_grid(params['batch_size']).to(DEVICE)

        samples = generator(noise)

        ############################
        # Update Discriminator network
        ###########################

        discriminator.zero_grad()

        real_score, real_h = discriminator(data)
        fake_score, fake_h = discriminator(samples.detach())

        dis_loss = torch.mean(criterion(real_score, torch.ones_like(real_score)) + criterion(
            fake_score, torch.zeros_like(fake_score)))
        dis_loss.backward()
        dis_optim.step()

        ############################
        # Update Generator network
        ###########################

        generator.zero_grad()
        fake_score, fake_h = discriminator(samples)
        og_gen_loss = criterion(fake_score, torch.ones_like(fake_score))
        gdpp_loss = compute_gdpp(fake_h.to('cpu'), real_h.to('cpu')).item()
        #         print(gdpp_loss)
        gen_loss = torch.mean(gdpp_loss + og_gen_loss)
        gen_loss.backward()
        gen_optim.step()

        if i % 5000 == 0:
            noise = dis.normal.Normal(torch.zeros(params['z_dim']), torch.ones(params['z_dim'])).sample(
                sample_shape=torch.tensor([params['number_evaluation_samples']])).to(DEVICE)
            samples = generator(noise).to('cpu').detach().numpy()
            data = data.to('cpu').detach().numpy()
            evaluate_samples(samples, data, model_num, i, is_ring_distribution=(model_num % 2 == 0))

    noise = dis.normal.Normal(torch.zeros(params['z_dim']), torch.ones(params['z_dim'])).sample(
        sample_shape=torch.tensor([params['number_evaluation_samples']])).to(DEVICE)
    samples = generator(noise).to('cpu').detach().numpy()
    data = data.to('cpu').detach().numpy()
    modes, high_ratio = evaluate_samples(samples, data, model_num, i + 1, is_ring_distribution=(model_num % 2 == 0))

    with open('./GDPP_%d.txt' % args.model, 'a') as f:
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
        max_iter=25000,
        gen_learning_rate=1e-3,
        disc_learning_rate=1e-4,
        number_evaluation_samples=2500,
    )

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=int, default=0, required=True)
    argparser.add_argument('--iter', type=int, default=25000)
    args = argparser.parse_args()

    run_gan(model_num=args.model)


