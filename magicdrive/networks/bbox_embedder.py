import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .embedder import get_embedder

XYZ_MIN = [-200, -300, -20]
XYZ_RANGE = [350, 650, 80]


def normalizer(mode, data):
    if mode == 'cxyz' or mode == 'all-xyz':
        # data in format of (N, 4, 3):
        mins = torch.as_tensor(
            XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == 'owhr':
        raise NotImplementedError(f"wait for implementation on {mode}")
    else:
        raise NotImplementedError(f"not support {mode}")
    return data


class ContinuousBBoxWithTextEmbedding(nn.Module):
    """
    Use continuous bbox corrdicate and text embedding with CLIP encoder
    """

    def __init__(
        self,
        n_classes,
        class_token_dim=768,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[768, 512, 512, 768],
        mode='cxyz',
        minmax_normalize=True,
        use_text_encoder_init=True,
        **kwargs,
    ):
        """
        Args:
            mode (str, optional): cxyz -> all points; all-xyz -> all points;
                owhr -> center, l, w, h, z-orientation.
        """
        super().__init__()

        self.mode = mode
        if self.mode == 'cxyz':
            input_dims = 3
            output_num = 4  # 4 points
        elif self.mode == 'all-xyz':
            input_dims = 3
            output_num = 8  # 8 points
        elif self.mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {mode}")
        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        logging.info(
            f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
            f"{self.fourier_embedder.out_dim} dims.")

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

        # for class token
        self._class_tokens_set_or_warned = not self.use_text_encoder_init
        if trainable_class_token:
            # parameter is trainable, buffer is not
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_parameter("_class_tokens", nn.Parameter(class_tokens))
        else:
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_buffer("_class_tokens", class_tokens)
            if not self.use_text_encoder_init:
                logging.warn(
                    "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not"
                    " trainable but you set `use_text_encoder_init` to False. "
                    "Please check your config!")

        # null embedding
        self.null_class_feature = torch.nn.Parameter(
            torch.zeros([class_token_dim]))
        self.null_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num]))

    @property
    def class_tokens(self):
        if not self._class_tokens_set_or_warned:
            logging.warn(
                "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not "
                "trainable and used without initialization. Please check your "
                "training code!")
            self._class_tokens_set_or_warned = True
        return self._class_tokens

    def prepare(self, cfg, **kwargs):
        if self.use_text_encoder_init:
            self.set_category_token(
                kwargs['tokenizer'], kwargs['text_encoder'],
                cfg.dataset.object_classes)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens "
                         "initilzed with random.")

    @torch.no_grad()
    def set_category_token(self, tokenizer, text_encoder, class_names):
        logging.info("[ContinuousBBoxWithTextEmbedding] Initialzing your "
                     "class_tokens with text_encoder")
        self._class_tokens_set_or_warned = True
        device = self.class_tokens.device
        for idx, name in enumerate(class_names):
            inputs = tokenizer(
                [name], padding='do_not_pad', return_tensors='pt')
            inputs = inputs.input_ids.to(device)
            # there are two outputs: last_hidden_state and pooler_output
            # we use the pooled version.
            hidden_state = text_encoder(inputs).pooler_output[0]  # 768
            self.class_tokens[idx].copy_(hidden_state)

    def add_n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        hidden_states = torch.cat([hidden_states, uncond_token], dim=1)
        return hidden_states

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, bboxes: torch.Tensor, classes: torch.LongTensor,
                masks=None, **kwargs):
        """Please do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, N, 4, 3) for cxyz mode.
            classes (torch.LongTensor): (B, N)

        Return:
            size B x N x emb_dim=768
        """
        (B, N) = classes.shape
        bboxes = rearrange(bboxes, 'b n ... -> (b n) ...')

        if masks is None:
            masks = torch.ones(len(bboxes))
        else:
            masks = masks.flatten()
        masks = masks.unsqueeze(-1).type_as(self.null_pos_feature)

        # box
        if self.minmax_normalize:
            bboxes = normalizer(self.mode, bboxes)
        pos_emb = self.fourier_embedder(bboxes)
        pos_emb = pos_emb.reshape(
            pos_emb.shape[0], -1).type_as(self.null_pos_feature)
        pos_emb = pos_emb * masks + self.null_pos_feature[None] * (1 - masks)

        # class
        cls_emb = torch.stack([self.class_tokens[i] for i in classes.flatten()])
        cls_emb = cls_emb * masks + self.null_class_feature[None] * (1 - masks)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)
        emb = rearrange(emb, '(b n) ... -> b n ...', n=N)
        return emb
