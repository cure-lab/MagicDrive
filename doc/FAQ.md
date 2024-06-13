### Q1: about `temp_attn_type`
This parameter controls the order of each module in the transformer block.
| name            | order                                                         | Mem     | Mem with ckpt | note (added params)       |
| --------------- | ------------------------------------------------------------ | -------- | ---- | ------------ |
| *ts_first_ff*   | **Temporal attention**，**ST-Attention**，self-attention，cross-attention，cross-view attention | OOM      | 18.6 | 118M         |
| *s_first_t_ff*  | **ST-Attention**，self-attention，cross-attention，cross-view attention，**Temporal attention**，ff | OOM      | 18.2 |              |
| *s_ff_t_last*   | **ST-Attention**，self-attention，cross-attention，cross-view attention，ff，**Temporal attention** | OOM      | 18.3 |              |
| *t_first_ff*    | **Temporal attention**，self-attention，cross-attention，cross-view attention | 28.7     |      | 59M          |
| *t_ff*          | self-attention，cross-attention，cross-view attention，**Temporal attention**，ff | 27.9     |      |              |
| *t_last*        | self-attention，cross-attention，cross-view attention，ff，**Temporal attention** | **27.5** |      |              |
| *_ts_first_ff*  | **Temporal attention**，**ST-self-Attention**，cross-attention，cross-view attention | 31.7     |      | 71M          |
| *_s_first_t_ff* | **ST-self-Attention**，cross-attention，cross-view attention，**Temporal attention**，ff | 31.2     |      |              |
| *_s_ff_t_last*  | **ST-self-Attention**，cross-attention，cross-view attention，ff，**Temporal attention** | **28.4** | 17.6 | tune-a-video |


Note:
1. Results are tested on single V100, which keep similar on multiple GPUs. 224x400 for each view, 7 frames.
2. `ff` is feed-forward.
3. ST-self-Attention: reuse the parameters in `self-attention`, only `q-projection` is trainable.
4. **Bold** are trainable. In the latest version, we also make `cross-view attention` trainable for video generation.
