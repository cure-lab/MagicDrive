import logging
import torch


class Embedder:
    """
    borrow from
    https://github.com/zju3dv/animatable_nerf/blob/master/lib/networks/embedder.py
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, num_freqs, include_input=True, log_sampling=True):
    embed_kwargs = {
        "input_dims": input_dims,
        "num_freqs": num_freqs,
        "max_freq_log2": num_freqs - 1,
        "include_input": include_input,
        "log_sampling": log_sampling,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    logging.debug(f"embedder out dim = {embedder_obj.out_dim}")
    return embedder_obj
