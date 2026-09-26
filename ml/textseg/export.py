"""Export a trained checkpoint to ONNX and check the contract with onnxruntime.

Usage:
    python ml/textseg/export.py --checkpoint <run>/best.pt --out <run>/textseg.onnx

Contract (see model.py): `images` [1, 3, H, W] 0..1 RGB -> `prob` [1, 1, H/4, W/4].
H and W are dynamic; the eval and the extension feed 1280x1280.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import OUTPUT_STRIDE, ExportWrapper, TextMapNet

SIZE = 1280


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = TextMapNet(state["backbone"], pretrained=False)
    net.load_state_dict(state["model"])
    wrapper = ExportWrapper(net).eval()

    dummy = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(
        wrapper, dummy, str(args.out), opset_version=args.opset,
        input_names=["images"], output_names=["prob"],
        dynamic_axes={"images": {2: "height", 3: "width"}, "prob": {2: "map_h", 3: "map_w"}},
        dynamo=False,
    )

    import onnxruntime as rt

    session = rt.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    out = session.run(None, {"images": dummy.numpy()})[0]
    expected = (1, 1, SIZE // OUTPUT_STRIDE, SIZE // OUTPUT_STRIDE)
    assert out.shape == expected, f"{out.shape} != {expected}"
    with torch.no_grad():
        reference = wrapper(dummy).numpy()
    drift = float(np.abs(out - reference).max())
    assert drift < 1e-3, f"onnx and torch disagree by {drift}"
    print(f"{args.out} {args.out.stat().st_size / 1e6:.1f} MB, prob {list(out.shape)}, max drift {drift:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
