"""Full-ring regression checks, also runnable with standard-library unittest."""

import contextlib
import io
import logging
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from pyrinst.cli import build_parser, fep_eval, geom, sampling
from pyrinst.geometries import HarmRef, Instanton, InstRef, Springs
from pyrinst.io.xyz import load, save
from pyrinst.opt.hessian import bfgs, bofill
from pyrinst.opt.optimizers import LBFGS, ModeFollowing, StreamBedWalk
from pyrinst.opt.projections import centroid
from pyrinst.potentials import CachedExecutor, Level, SingleExecutor
from pyrinst.utils.numderiv import grad_from_energy, hess_from_grad
from pyrinst.utils.pimc import InstFEP
from pyrinst.utils.units import EV


class QuadraticExecutor:
    def compute(self, data, level=Level.FREQ):
        data.V = 0.5 * np.sum(data.x**2, axis=(1, 2))
        data.G = data.x.copy()
        data.H = np.tile(np.eye(data.dof), (len(data.x), 1, 1)) if level == Level.FREQ else None


def quadratic_potential(x, level=Level.FREQ):
    return 0.5 * np.sum(x**2), x.copy(), np.eye(x.size) if level == Level.FREQ else None


class FullRingTests(unittest.TestCase):
    def setUp(self):  # noqa: N802 - unittest lifecycle hook
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous)

    def reference(self):
        ref = HarmRef(np.zeros((2, 3)), symbols=["H", "H"])
        ref.energy = 0.0
        ref.hess = np.eye(6)
        ref.calc_freq()
        return ref

    def path(self, n=4, full=True):
        ref = self.reference()
        x = np.random.default_rng(42).normal(scale=0.05, size=(n, 2, 3))
        x -= x.mean(axis=0)
        inst = InstRef(x, ref.symbols, masses=ref.m, beta=100, full_ring=full, links=[ref])
        QuadraticExecutor().compute(inst)
        return inst

    def test_spring_and_total_derivatives(self):
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                x = np.random.default_rng(n).normal(size=(n, 2, 3))
                spring = Springs(n, 3.7, np.array([1.0, 2.0]))
                assert_allclose(spring.gradient(x), grad_from_energy(x, spring.potential, 1e-5), atol=1e-8)
                assert_allclose(spring.hessian(x), hess_from_grad(x, spring.gradient, 1e-5), atol=1e-8)
                inst = Instanton(x, masses=spring.masses, beta=3.7, full_ring=True)
                QuadraticExecutor().compute(inst)
                def energy(coords, spring=spring):
                    return 0.5 * np.sum(coords**2) + spring.potential(coords)
                assert_allclose(inst.G, grad_from_energy(x, energy, 1e-5), atol=1e-8)
                assert_allclose(inst.H, hess_from_grad(x, lambda y, s=spring: y + s.gradient(y), 1e-5), atol=1e-8)
                self.assertTrue(np.any(inst.H[:6, -6:]))

    def test_half_full_equivalence(self):
        for n in (1, 2, 4):
            with self.subTest(n=n):
                half = self.path(n, full=False)
                full = half.to_full_ring()
                for name in ("V", "S", "E", "BN"):
                    assert_allclose(getattr(half, name), getattr(full, name), atol=1e-10)
                mapping = np.concatenate((np.eye(half.x.size), np.eye(half.x.size).reshape(
                    n, 6, half.x.size)[::-1].reshape(half.x.size, half.x.size)))
                assert_allclose(half.G.ravel(), mapping.T @ full.G.ravel(), atol=1e-10)
                assert_allclose(half.H, mapping.T @ full.H @ mapping, atol=1e-10)
                assert_allclose(half.hessian_full(), full.hessian_full(), atol=1e-10)

    def test_pickle_and_conversion(self):
        half = self.path(full=False)
        state = half.__getstate__()
        del state[1]["full_ring"]
        legacy = InstRef.__new__(InstRef)
        legacy.__setstate__(state)
        self.assertFalse(legacy.full_ring)
        del legacy.full_ring
        legacy = pickle.loads(pickle.dumps(legacy))
        self.assertFalse(legacy.full_ring)
        legacy = legacy.to_full_ring()
        restored = pickle.loads(pickle.dumps(legacy))
        self.assertTrue(restored.full_ring)
        self.assertEqual(restored.N, len(restored.x))
        assert_allclose(restored.x, np.concatenate((half.x, half.x[::-1])))
        self.assertIsNone(restored.modes)

    def test_whole_object_conversion(self):
        half = self.path(full=False)
        half.freqs = np.arange(half.N * half.dof, dtype=float)
        half.modes = np.eye(half.N * half.dof)
        half.harm_energies = np.array([0.1, 0.2])
        full = half.to_full_ring()
        self.assertIs(type(full), type(half))
        self.assertFalse(half.full_ring)
        self.assertTrue(full.full_ring)
        self.assertEqual(full.N, half.N)
        self.assertEqual(full.beta, half.beta)
        self.assertEqual(full.T, half.T)
        for name in ("coords", "energy", "grad", "hess"):
            value = getattr(half, name)
            assert_allclose(getattr(full, name), np.concatenate((value, value[::-1])))
            self.assertFalse(np.shares_memory(getattr(full, name), value))
        for name in ("freqs", "modes", "harm_energies", "masses"):
            assert_allclose(getattr(full, name), getattr(half, name))
            self.assertFalse(np.shares_memory(getattr(full, name), getattr(half, name)))
        self.assertIsNot(full.links[0], half.links[0])
        full.validate_reference()
        second = full.to_full_ring()
        assert_allclose(second.x, full.x)
        self.assertFalse(np.shares_memory(second.x, full.x))
        half.hess = half.H
        self.assertIsNone(half.to_full_ring().hess)
        with self.assertRaisesRegex(ValueError, "bead Hessians"):
            half.hessian_full()

    def test_hessian_updates(self):
        hess = np.diag([2.0, 3.0])
        step = np.array([0.1, -0.2])
        for update in (bfgs, bofill):
            assert_allclose(update(hess.copy(), step, hess @ step), hess, atol=1e-14)
            gradient_change = np.array([0.4, -0.5])
            updated = update(hess.copy(), step, gradient_change)
            assert_allclose(updated @ step, gradient_change, atol=1e-14)
            assert_allclose(updated, updated.T, atol=1e-14)
        # A nonzero residual orthogonal to the step must not divide by zero.
        step = np.array([1.0, 0.0])
        updated = bofill(hess.copy(), step, np.array([2.0, 1.0]))
        assert_allclose(updated @ step, [2.0, 1.0])

    def test_constrained_derivatives(self):
        inst = self.path(n=3)
        p = centroid(inst.x).reshape(6, -1)
        projector = np.eye(inst.x.size) - p.T @ p
        initial = inst.x.copy()

        def energy(displacement):
            inst.x = initial + (projector @ displacement.ravel()).reshape(initial.shape)
            QuadraticExecutor().compute(inst)
            return inst.V

        def gradient(displacement):
            energy(displacement)
            return inst.G

        expected_grad, expected_hess = inst.G.copy(), inst.H.copy()
        assert_allclose(grad_from_energy(np.zeros_like(initial), energy, 1e-5), expected_grad, atol=1e-8)
        assert_allclose(hess_from_grad(np.zeros_like(initial), gradient, 1e-5), expected_hess, atol=1e-8)

    def test_invalid_reference_and_counts(self):
        inst = self.path()
        inst.links[0].x += 0.1
        with self.assertRaisesRegex(ValueError, "centroid"):
            inst.validate_reference()
        inst.links = []
        with self.assertRaisesRegex(ValueError, "HarmRef"):
            inst.validate_reference()
        for n in (0, 1):
            with self.assertRaises(ValueError):
                inst.interpolate(n)
        with self.assertRaises(ValueError):
            InstFEP(inst, nbeads=inst.N + 1)

    def test_initial_guess_and_interpolation(self):
        ref = self.reference()
        for full, n in ((False, 2), (False, 6), (True, 2), (True, 5)):
            with self.subTest(full=full, n=n):
                inst = ref.get_inst_guess(n, 100, full_ring=full)
                self.assertEqual(len(inst.x), n if full else n // 2)
                assert_allclose(inst.x.mean(axis=0), ref.x, atol=1e-15)
                QuadraticExecutor().compute(inst)
                inst.interpolate(7 if full else 8)
                assert_allclose(inst.x.mean(axis=0), ref.x, atol=1e-15)
                self.assertIsNone(inst.hess)
                self.assertIsNone(inst.harm_energies)
        with self.assertRaises(ValueError):
            ref.get_inst_guess(3, 100)

    def test_optimization_preserves_centroid(self):
        for optimizer in (ModeFollowing, StreamBedWalk, LBFGS):
            for project in (False, True):
                for update in (False, True):
                    with self.subTest(optimizer=optimizer, project=project, update=update):
                        inst = self.path(n=5)
                        initial = inst.x.mean(axis=0).copy()
                        opt = optimizer(order=0, executor=QuadraticExecutor(), maxstep=0.05,
                                        project=project, update=update)
                        centers = []
                        opt.search(inst, gtol=1e-8, maxiter=100,
                                   callback=lambda data, centers=centers: centers.append(data.x.mean(axis=0)))
                        self.assertLess(np.linalg.norm(inst.G), 1e-8)
                        for center in centers:
                            assert_allclose(center, initial, atol=1e-14)
                        assert_allclose(inst.H @ centroid(inst.x).reshape(6, -1).T, 0, atol=1e-12)

    def test_cli_flags_and_xyz(self):
        parser = build_parser()
        args = parser.parse_args(["geom", "path.xyz", "--mode", "centroid", "--full-ring", "--beta", "100"])
        self.assertTrue(args.full_ring)
        self.assertFalse(parser.parse_args(["geom", "ref.pkl", "--mode", "centroid"]).full_ring)
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder) / "path.xyz")
            inst = self.path(n=5)
            save(filename, inst.x, inst.symbols)
            data, symbols, x, ext = geom.load_input_geometry(filename)
            result = geom.make_initial_geometry(args, data, symbols, x, ext, None)
            self.assertTrue(result.full_ring)
            self.assertEqual(result.N, 5)
            assert_allclose(result.x, inst.x, atol=1e-9)
            args.full_ring = False
            with self.assertRaises(ValueError):
                geom.make_initial_geometry(args, data, symbols, x, ext, None)
            save(filename, inst.x[:1], inst.symbols)
            save(filename, inst.x[:1], ["H", "D"], append=True)
            with self.assertRaises(OSError):
                geom.load_input_geometry(filename)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["geom", "ref.pkl", "--mode", "centroid", "--full-ring", "true"])

    def test_sampling_equivalence(self):
        with tempfile.TemporaryDirectory() as folder:
            half = self.path(full=False)
            full = half.to_full_ring()
            results = []
            for inst in (half, full):
                inst.final_output(str(Path(folder) / "inst"))
                polymer = InstFEP(inst, nbeads=inst.N)
                sampled = polymer.sample_normal_modes(8, sampler="numpy", rng=np.random.default_rng(7))
                coords = polymer.get_cart_pos(sampled)
                assert_allclose(coords.mean(axis=1), np.broadcast_to(inst.x.mean(axis=0), (8, 2, 3)), atol=1e-12)
                inst.freqs = polymer.freq_rp
                results.append((coords, polymer.harm_energies, inst.delta_free_energy()))
            for a, b in zip(results[0], results[1], strict=True):
                assert_allclose(a, b, atol=1e-10)

    def test_cli_optimize_resume_sample_fep(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ref = self.reference()
            ref.save(str(root / "ref"))
            for extra in ([], ["--full-ring"]):
                prefix = str(root / ("full" if extra else "half"))
                args = parser.parse_args(["geom", str(root / "ref.pkl"), "--mode", "centroid", "--beta", "100",
                                          "--nbeads", "4", "--maxiter", "100", "-g", "1e-10", "-o", prefix, *extra])
                executor = CachedExecutor(SingleExecutor(quadratic_potential))
                with patch.object(geom, "build_optimization_executor", return_value=executor):
                    geom.run(args, parser)
                    resume = parser.parse_args(["geom", prefix + ".pkl", "--mode", "centroid",
                                                "--maxiter", "0", "-o", prefix])
                    geom.run(resume, parser)
                with open(prefix + ".pkl", "rb") as handle:
                    inst = pickle.load(handle)
                self.assertEqual(inst.full_ring, bool(extra))
                sample_args = parser.parse_args(["sample", prefix + ".pkl", "-T", str(inst.T), "-N", "8",
                                                 "--nbeads", "4", "--nprandom", "-o", prefix + "_sample"])
                sampling.run(sample_args)
                for bead in range(4):
                    filename = prefix + f"_sample_{bead}.xyz"
                    symbols, coords, _ = load(filename, energy_pattern=False)
                    energies = 0.5 * np.sum(coords**2, axis=(1, 2)) / EV
                    save(filename, coords, symbols, [f"energy={energy:.16e}" for energy in energies])
                args = parser.parse_args(["fep-eval", prefix + ".pkl", "--nbeads", "4",
                                          "--prefix", prefix + "_sample"])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    fep_eval.run(args)
                self.assertIn("Delta F", output.getvalue())
                self.assertNotIn("nan", output.getvalue().lower())
                self.assertNotIn("inf", output.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
