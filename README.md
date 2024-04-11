# HGRN2: Gated Linear RNNs with State Expansion

Standalone code for hgrn2. For 1d, the input shape is (n, b, d), where n is sequence length, b is batch size and d is feature dim; For 2d, the input shape is (h, w, b, d).

## Hf model example
We also provide the implementations of models that are compatible with Transformers library:
```
from hgru2_pytorch import Hgrn2Config

config = Hgrn2Config()
model = AutoModel.from_config(config)
print(model)
```
This should return:
```
Hgrn2Model(
  (lower_bounds): Tensor((24, 1024), requires_grad=True)
  (embed_tokens): Embedding(50272, 1024, padding_idx=1)
  (layers): ModuleList(
    (0-23): 24 x Hgrn2DecoderLayer(
      (token_mixer): Hgru2(
        (in_proj): Linear(in_features=1024, out_features=3072, bias=False)
        (out_proj): Linear(in_features=1024, out_features=1024, bias=False)
        (norm): SimpleRMSNorm()
      )
      (token_norm): SimpleRMSNorm()
      (channel_mixer): GLU(
        (l1): Linear(in_features=1024, out_features=2816, bias=False)
        (l2): Linear(in_features=1024, out_features=2816, bias=False)
        (l3): Linear(in_features=2816, out_features=1024, bias=False)
      )
      (channel_norm): SimpleRMSNorm()
    )
  )
  (final_norm): SimpleRMSNorm()
)
```
