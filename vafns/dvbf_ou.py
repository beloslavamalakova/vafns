import torch
import torch.nn as nn
import math


def get_transition_model(transition_model, **kwargs):
    """Returns TransitionModel object."""
    transition_model = transition_model.split("_")
    transition_model = [s.capitalize() for s in transition_model]
    transition_model = "".join(transition_model)
    return eval("{}TransitionModel".format(transition_model))

# takes the transition model as a string, in the general case its divided by underscores which are reduced by the dict; ensures that below the transition model could be called without signs such as '_', and as a together written sequence of the two words 'Transition' and 'Model'

# kwargs is used for keyword arguments

class BayesFilter(nn.Module):
    def __init__(
        self,
            transition_model, #parametrizes transition f(z_t, u_t, /betta_t)
        noise_dim, # dim of w_t
        action_dim, #dim of u_t
        latent_dim, #dim of z_t; the number of nodes used
        input_dim, #dim of x_t
        hidden_size,
        kl_weight,
        annealing_steps=100,
    ):

        super().__init__()
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.kl_weight = kl_weight

        self.transition_model = transition_model

        # nn.Sequential MLP w/ relu
        # input is self.x /observations/
        self.extractor = nn.Sequential( #extractor -- from previous to the next tensor; sequential- the next layer takes the data from the previous
            nn.Linear(input_dim, hidden_size), #creates a single layer deep forward network
            nn.ReLU(), #positive outputs-- values; takes the maximal value between 0 and x
            nn.Linear(hidden_size, latent_dim),
        )

        self.inference = nn.Sequential(
            nn.Linear(noise_dim + latent_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * noise_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

        self.initial_net = nn.LSTM( #ensuring a starting point after propagating through data
            input_size=latent_dim,
            hidden_size=hidden_size,
            bidirectional=True, #connects two opposite directions to the same output
            dropout=0.25, #prevents overfitting; temporarily removes neurons from the net; injects some noise
        )
        self.initial_affine = nn.Linear(2 * hidden_size, 2 * noise_dim) #maps from the hidden size to the noise dim

        self.initial_noise_to_latent = nn.Sequential(
            nn.Linear(noise_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )

        self.anneal_steps = annealing_steps
        if self.anneal_steps != 0:
            self.anneal_rate = 1e-3
            self.update_annealing = self._update_annealing

        else:
            self.anneal_rate = 1.
            self.update_annealing = lambda *args: None

    def _update_annealing(self):
        if self.anneal_rate > 1.0 - 1e-5:
            self.anneal_rate = 1.0
        else:
            self.anneal_rate += 1.0 / self.anneal_steps

    def forward(self, observations, actions, logger=None): #computes output tensors from input tensors; the batches and seq_len to the extractor
        seq_len, batch_size = observations.shape[:2]

        transition_parameters = self.extractor(
            observations.view(-1, observations.shape[2:])
        ).view(seq_len, batch_size, -1)

        initial_w, _ = self.initial_net(transition_parameters) #bidirectional lstm takes bettas resulting in the first layer of w(noise)
        initial_w = initial_w.view(seq_len, batch_size, 2, -1) #outputs seq_len*batch_size; scans a tensor and the backpropagates through it
        initial_w = torch.cat([initial_w[-1, :, 0], initial_w[0, :, 1]], dim=-1)
        initial_w = self.initial_affine(initial_w)

        mu, logstd = initial_w.split(2, dim=-1) #logarithm of std
        w_t = self.generate_samples(mu, logstd)
        z_t = self.initial_noise_to_latent(w_t)

        latents = [z_t] #latent_dim makes a list with a given first element-- latent
        dists = [initial_w] #dist of distribution
        for t in range(1, seq_len):
            noise_dist = self.inferece(
                torch.cat([transition_parameters[t], z_t, u[t]], dim=-1) #into 1 dimension
            )
            dists.append(noise_dist) #noise distribution
            w_t = self.generate_samples(*noise_dist.split(2, dim=-1)) #as a given input
            z_t = self.transition_model(z_t, actions[t], w_t)
            latents.append(z_t)

        latents = torch.stack(latents, dim=0) #taking the lists -> new dimension
        dists = torch.stack(dists, dim=0)

        observation_preds = self.decoder(latents.view(seq_len * batch_size, -1)).view(
            *observations.shape
        )
        rec_loss = nn.functional.mse_loss(observation_preds, observations)

        mu, logstd = dists.split(2, dim=-1)
        kl_loss = -logstd + self.anneal_rate * (
            torch.exp(2 * logstd).clamp(1e-5) + mu ** 2
        ) #clamp-- the first arg is the min, the second is the max; taking the exp of logstd
        kl_loss = kl_loss.mean()

        loss = rec_loss + self.kl_weight * kl_loss

        if logger is not None:
            logger["loss/loss"].append(loss.item())
            logger["loss/kl_loss"].append(kl_loss.item())
            logger["loss/rec_loss"].append(rec_loss.item())

    def generate_samples(self, mu, logstd):
        std = torch.exp(logstd).clapm(1e-6) #1*10^-6
        samples = torch.randn_like(mu)

        return samples * std + mu


class TransitionModel(nn.Module): #abstract class for Transition Models; all should have latent, action, noise dims
    def __init__(self, latent_dim, action_dim, noise_dim, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim

    def forward(self, latent, action, noise):
        """
        Implements transition in latent space.
        Args:
             latent : z_t
             action : u_t
             noise : w_t
        Returns:
            Next latent z_{t+1}.
        """
        pass


    # GENERAL NOTES: torch; process is in the latent space; linear layer mapping latent/action/noise to the hidden size
    # 1) parametrizing over more than an x;
    # 2) embedding betta,
    # 3) ornstein-uhlenbeck process - stochastic ODE Visicek (OU) process - using mu as a 0
    # 4) hidden size to latent dim
    # 5) return t+1 next state

    # POINTS UNSURE ABOUT
    # what kind of function should the ornstein_uhlenbeck be?
    # what is betta representing in the dvbf
    # should the parametrization be done through alpha betta and gamma, and how is it connected to the value of betta in the dvbf
    # change of the values -- sigma, mu, theta
    # initialization of the parameters, should it be done in the __init__ or in the def forward function


class OrnsteinUhlenbeckTransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim, noise_dim, betta, gamma, alpha, ornstein_uhlenbeck, hidden_size=16, mu=0, theta= 0.15, sigma = 0.2, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.ornstein_uhlenbeck = ornstein_uhlenbeck

    def forward(self, latent, action, noise):

        self.alpha = nn.Parameter(
            torch.rndn(latent_dim) * self.mu)

        self.betta = nn.Parameter(
            torch.rndn(action_dim) * self.mu)

        self.gamma = nn.Parameter(
            torch.rndn(noise_dim) * self.mu)

        x = self.Linear(aplha + betta + gamma, ornstein_uhlenbech)

        def  ornstein_uhlenbeck(self):
            dx = self.theta * (self.mu - self.x)
            dx = dx + self.sigma * torch.rndn(len(self.x))
            self.x = self.x + dx
            return self.x

        self.net = Sequential(
            nn.Linear(latent_dim + action_dim + noise_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
            nn.Softmax(),
        )


    @staticmethod
    def _get_output(matrix, weight, feature):
        # matrix [num_matrices, dim1, dim2]
        # weight [batch, num_matrices]
        # feature [batch, dim1]
        num_matrices, dim1, dim2 = matrix.size()
        feature_matrix = torch.mm(weight, matrix.view(num_matrices, -1)).view(
            -1, dim1, dim2
        )
        return torch.sum(feature_matrix * feature.unsqueeze(-1), dim=1)

    def forward(self, latent, action, noise):
        weights = self.net(torch.cat([latent, action], dim=-1))

        latent_out = self._get_output(self.latent_matrix, weights, latent)
        action_out = self._get_output(self.action_matrix, weights, action)
        noise_out = self._get_output(self.noise_matrix, weights, noise)

        return latent_out + action_out + noise_out