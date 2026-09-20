from __future__ import annotations

import unittest

import torch

from causal_codec import CausalCodec, HOP_LENGTH, architecture_stats


class CausalCodecTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.codec = CausalCodec(latent_dim=4, channels=(4, 8, 12, 16, 24))
        self.codec.eval()

    def test_shape(self) -> None:
        wav = torch.randn(2, 1, 3 * HOP_LENGTH)
        out, z = self.codec(wav)
        self.assertEqual(z.shape, (2, 4, 3))
        self.assertEqual(out.shape, wav.shape)

    def test_decoder_full_step_parity(self) -> None:
        z = torch.randn(1, 4, 4)
        with torch.inference_mode():
            full = self.codec.decode(z)
            step = self.codec.decoder.stream().decode_chunk(z)
        self.assertLessEqual(float((full - step).abs().max()), 1e-6)

    def test_decoder_future_invariance(self) -> None:
        z0 = torch.randn(1, 4, 5)
        z1 = z0.clone()
        z1[..., 3:] = torch.randn_like(z1[..., 3:])
        with torch.inference_mode():
            y0 = self.codec.decode(z0)
            y1 = self.codec.decode(z1)
        self.assertTrue(torch.equal(y0[..., :3 * HOP_LENGTH], y1[..., :3 * HOP_LENGTH]))

    def test_encoder_future_invariance(self) -> None:
        x0 = torch.randn(1, 1, 5 * HOP_LENGTH)
        x1 = x0.clone()
        x1[..., 3 * HOP_LENGTH:] = torch.randn_like(x1[..., 3 * HOP_LENGTH:])
        with torch.inference_mode():
            z0 = self.codec.encode(x0)
            z1 = self.codec.encode(x1)
        self.assertTrue(torch.equal(z0[..., :3], z1[..., :3]))

    def test_stream_reset(self) -> None:
        z = torch.randn(1, 4, 3)
        stream = self.codec.decoder.stream()
        with torch.inference_mode():
            y0 = stream.decode_chunk(z)
            stream.reset()
            y1 = stream.decode_chunk(z)
        self.assertTrue(torch.equal(y0, y1))

    def test_default_architecture_contract(self) -> None:
        codec = CausalCodec()
        stats = architecture_stats(codec)
        self.assertEqual(codec.encoder.hop_length, HOP_LENGTH)
        self.assertEqual(codec.decoder.hop_length, HOP_LENGTH)
        self.assertLess(stats.total_parameters, 35_000_000)
        keys = codec.state_dict().keys()
        self.assertIn("encoder.pre.weight", keys)
        self.assertIn("decoder.stages.0.up.weight", keys)
        self.assertIn("decoder.post.weight", keys)


if __name__ == "__main__":
    unittest.main()
