"""Thin wrapper around the Rockchip NPU runtime (rknn-toolkit-lite2).

The RK3576 exposes two NPU cores, so the default core mask is
``NPU_CORE_0_1`` (the three core mask that is used on other platforms is
rejected by the driver with a warning).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RknnError(RuntimeError):
    """Raised when the NPU runtime reports an error."""


class RknnModel:
    """A compiled .rknn model running on the Rockchip NPU."""

    def __init__(
        self,
        model_path: str | Path,
        core_mask: int | None = None,
        batch_size: int | None = 2,
    ) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RknnError(
                "rknn-toolkit-lite2 is not installed.\n"
                "Install it with:  pip install rknn-toolkit-lite2==2.3.2"
            ) from exc

        self.model_path = Path(model_path)
        self.core_mask = core_mask if core_mask is not None else RKNNLite.NPU_CORE_0_1
        self.batch_size = batch_size

        self.rknn = RKNNLite()
        rc = self.rknn.load_rknn(str(self.model_path))
        if rc != 0:
            raise RknnError(f"load_rknn failed with code {rc}: {self.model_path}")
        rc = self.rknn.init_runtime(core_mask=self.core_mask)
        if rc != 0:
            raise RknnError(f"init_runtime failed with code {rc}")

    def infer(self, images: np.ndarray) -> list[np.ndarray]:
        """Run the model on a batch of NHWC uint8 images.

        Parameters
        ----------
        images
            Batch of shape (N, H, W, C) or a single (H, W, C) image.

        The compiled model has a fixed batch size (both eyes in one
        call).  Larger inputs are split into chunks of that size and the
        results are concatenated again.
        """
        images = np.ascontiguousarray(images, dtype=np.uint8)
        if images.ndim == 3:
            images = images[None]
        count = images.shape[0]

        size = self.batch_size or count
        chunks = []
        for start in range(0, count, size):
            chunk = images[start : start + size]
            if chunk.shape[0] < size:
                chunk = np.concatenate(
                    [chunk, np.repeat(chunk[-1:], size - chunk.shape[0], axis=0)],
                    axis=0,
                )
            chunks.append(chunk)

        collected: list[list[np.ndarray]] | None = None
        for chunk in chunks:
            outputs = self.rknn.inference(inputs=[chunk])
            if outputs is None:
                raise RknnError("NPU inference returned no data")
            outputs = [np.asarray(o) for o in outputs]
            if collected is None:
                collected = [[] for _ in outputs]
            for acc, out in zip(collected, outputs):
                take = min(out.shape[0], chunk.shape[0])
                acc.append(out[:take])

        assert collected is not None
        return [np.concatenate(acc, axis=0)[:count] for acc in collected]

    def close(self) -> None:
        if getattr(self, "rknn", None) is not None:
            self.rknn.release()
            self.rknn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
