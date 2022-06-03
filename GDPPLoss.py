import torch
import torch.nn.functional as f

def compute_gdpp(phi_fake, phi_real):
    def compute_diversity(phi):
        phi = f.normalize(phi, p=2, dim=1)
        S_B = torch.mm(phi, phi.t())
        eig_vals, eig_vecs = torch.symeig(S_B, eigenvectors=True)
        return eig_vals, eig_vecs

    def normalize_min_max(eig_vals):
        min_v, max_v = torch.min(eig_vals), torch.max(eig_vals)
        return (eig_vals - min_v) / (max_v - min_v)

    fake_eig_vals, fake_eig_vecs = compute_diversity(phi_fake)
    real_eig_vals, real_eig_vecs = compute_diversity(phi_real)
    # Scaling factor to make the two losses operating in comparable ranges.
    magnitude_loss = 0.0001 * f.mse_loss(target=real_eig_vals, input=fake_eig_vals)
    structure_loss = -torch.sum(torch.mul(fake_eig_vecs, real_eig_vecs), 0)
    normalized_real_eig_vals = normalize_min_max(real_eig_vals)
    weighted_structure_loss = torch.sum(torch.mul(normalized_real_eig_vals, structure_loss))
    return magnitude_loss + weighted_structure_loss