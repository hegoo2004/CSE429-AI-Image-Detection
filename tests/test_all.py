"""
test_all.py — Full project test suite (21 tests, no internet/GPU required)
===========================================================================
Run: python test_all.py
     pytest test_all.py -v
"""

import sys, unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")


# ── Mock backbone (no open_clip download needed) ──────────────────────────────

def _mock_backbone(embed_dim=768):
    class Mock(nn.Module):
        EMBED_DIM = embed_dim
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3*224*224, embed_dim, bias=False)
            for p in self.proj.parameters(): p.requires_grad = False
            self.proj.eval()
        def train(self, mode=True):
            super().train(mode); self.proj.eval(); self.training = False; return self
        def forward(self, x):
            return F.normalize(self.proj(x.flatten(1)), dim=-1)
    return Mock()


def _patched(cls, *args, **kwargs):
    import model as _m
    orig = _m.CLIPBackbone
    _m.CLIPBackbone = lambda: _mock_backbone()
    obj = cls(*args, **kwargs)
    _m.CLIPBackbone = orig
    return obj


def dummy(B=4): return torch.randn(B, 3, 224, 224)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMockBackbone(unittest.TestCase):
    def setUp(self): self.bb = _mock_backbone()

    def test_output_shape(self):
        self.assertEqual(self.bb(dummy()).shape, (4, 768))

    def test_l2_normalised(self):
        out = self.bb(dummy())
        self.assertTrue(torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5))

    def test_frozen(self):
        for p in self.bb.parameters(): self.assertFalse(p.requires_grad)


class TestLinearProbingDetector(unittest.TestCase):
    def setUp(self):
        from model import LinearProbingDetector
        self.model = _patched(LinearProbingDetector)
        self.model.eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(dummy(4))
        self.assertEqual(out.shape, (4, 1))

    def test_trainable_params_769(self):
        counts = self.model.param_count()
        self.assertEqual(counts["trainable"], 769)

    def test_gradients_only_in_head(self):
        self.model.train()
        loss = self.model(dummy(2)).sum()
        loss.backward()
        for n, p in self.model.head.named_parameters():
            self.assertIsNotNone(p.grad, f"Head '{n}' missing gradient")
        for n, p in self.model.backbone.named_parameters():
            self.assertIsNone(p.grad, f"Backbone '{n}' should have no gradient")

    def test_predict_proba_range(self):
        with torch.no_grad():
            p = self.model.predict_proba(dummy(8))
        self.assertEqual(p.shape, (8,))
        self.assertTrue((p >= 0).all() and (p <= 1).all())

    def test_predict_binary(self):
        with torch.no_grad():
            preds = self.model.predict(dummy(8))
        self.assertTrue(set(preds.tolist()).issubset({0, 1}))

    def test_batch_size_1(self):
        with torch.no_grad():
            out = self.model(dummy(1))
        self.assertEqual(out.shape, (1, 1))

    def test_batch_size_32(self):
        with torch.no_grad():
            out = self.model(dummy(32))
        self.assertEqual(out.shape, (32, 1))

    def test_output_dtype_float32(self):
        with torch.no_grad():
            out = self.model(dummy(4))
        self.assertEqual(out.dtype, torch.float32)

    def test_backbone_stays_eval_in_train_mode(self):
        self.model.train()
        self.assertFalse(self.model.backbone.training)

    def test_embed_dim_matches_backbone(self):
        w = self.model.head[-1].weight.shape
        self.assertEqual(w[1], self.model.backbone.EMBED_DIM)


class TestNearestNeighbourDetector(unittest.TestCase):
    def setUp(self):
        from model import NearestNeighbourDetector
        self.model = _patched(NearestNeighbourDetector)
        self._populate()

    def _populate(self, n=50):
        self.model.real_bank = F.normalize(torch.randn(n, 768), dim=-1)
        self.model.fake_bank = F.normalize(torch.randn(n, 768), dim=-1)
        self.model._bank_built = True

    def test_raises_before_bank_built(self):
        from model import NearestNeighbourDetector
        m = _patched(NearestNeighbourDetector)
        with self.assertRaises(RuntimeError):
            m.forward(dummy(2))

    def test_predict_shape(self):
        with torch.no_grad():
            preds = self.model.predict(dummy(4))
        self.assertEqual(preds.shape, (4,))

    def test_predict_valid_labels(self):
        with torch.no_grad():
            preds = self.model.predict(dummy(4))
        self.assertTrue(set(preds.tolist()).issubset({0, 1}))

    def test_score_shape(self):
        with torch.no_grad():
            scores = self.model.forward(dummy(6))
        self.assertEqual(scores.shape, (6,))

    def test_batch_size_1(self):
        with torch.no_grad():
            preds = self.model.predict(dummy(1))
        self.assertEqual(preds.shape, (1,))


class TestBuildModelFactory(unittest.TestCase):
    def _build(self, variant):
        import model as _m
        orig = _m.CLIPBackbone
        _m.CLIPBackbone = lambda: _mock_backbone()
        m = _m.build_model(variant)
        _m.CLIPBackbone = orig
        return m

    def test_linear_variant(self):
        from model import LinearProbingDetector
        self.assertIsInstance(self._build("linear"), LinearProbingDetector)

    def test_nn_variant(self):
        from model import NearestNeighbourDetector
        self.assertIsInstance(self._build("nn"), NearestNeighbourDetector)

    def test_invalid_raises(self):
        from model import build_model
        with self.assertRaises(ValueError):
            build_model("invalid")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestMockBackbone, TestLinearProbingDetector,
                TestNearestNeighbourDetector, TestBuildModelFactory]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"\n{'='*50}")
    if result.wasSuccessful():
        print(f"✓ All {result.testsRun} tests passed!")
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
    sys.exit(0 if result.wasSuccessful() else 1)
