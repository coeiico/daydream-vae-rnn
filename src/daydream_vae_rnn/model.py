"""
The model for daydream-vae-rnn.

a convolutional VAE compresses frames into a latent vector; an MDN-RNN (an LSTM
predicting a mixture of Gaussians) predicts the next latent; a tiny linear controller
is trained with CMA-ES. Reuse VAE code from knobs-vae.

forward() accepts return_cache=True and returns a dict of internal activations, and a
patch argument (or forward hooks) overwrites a named activation for interventions.
"""

# TODO: implement. See README.md for the plan.
