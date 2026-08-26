from __future__ import annotations

import ast
import json
import os
import inspect
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

import cf_ca_interaction as interaction


HERE = Path(__file__).resolve().parent


def synthetic_sample(size: int = 4096, seed: int = 123) -> interaction.LocalGeometrySample:
    rng = np.random.default_rng(seed)
    return interaction.LocalGeometrySample(
        kappa=rng.uniform(0.15, 1.6, size),
        kappa_prime=rng.normal(0.0, 0.7, size),
        kappa_tau=rng.normal(0.0, 0.8, size),
        weights=np.ones(size),
        k_eff=1.0,
    )


def straight_trace(offset: float = 0.0) -> interaction.ContourTrace:
    spacing = 0.1
    x = np.arange(0.0, 3.0, spacing)
    points = np.column_stack((x, np.full_like(x, offset), np.zeros_like(x)))
    tangent = np.tile([1.0, 0.0, 0.0], (len(x), 1))
    zeros = np.zeros_like(points)
    return interaction.ContourTrace(
        points=(points, points + np.array([0.0, 0.0, 1.0])),
        tangents=(tangent, tangent.copy()),
        r2=(zeros, zeros.copy()),
        r3=(zeros.copy(), zeros.copy()),
        k_eff=1.0,
        q_spacing=spacing,
    )


class SpectrumAndGeometryTests(unittest.TestCase):
    def test_reference_condition_can_be_zero_or_highest_salt(self) -> None:
        spectra=[interaction.FitSpectrum(value,1.0+value/100,1.0,0.2,
                 Path(f"{value:g}.csv")) for value in (0.0,10.0,100.0)]
        self.assertEqual(interaction.select_reference_spectrum(
            spectra,"zero").concentration_mM,0.0)
        self.assertEqual(interaction.select_reference_spectrum(
            spectra,"highest").concentration_mM,100.0)
        with self.assertRaisesRegex(ValueError,"zero-salt"):
            interaction.select_reference_spectrum(spectra[1:],"zero")
        with self.assertRaisesRegex(ValueError,"zero.*highest"):
            interaction.select_reference_spectrum(spectra,"middle")

    def test_gamma_second_moment_is_k_eff_squared(self) -> None:
        for width in (0.1, 0.2, 0.5):
            self.assertAlmostEqual(interaction.gamma_radial_moment(2, width, 1.7), 1.7**2, places=12)

    def test_conditional_covariance_is_symmetric_psd(self) -> None:
        covariance = interaction.conditional_derivative_covariance(0.22)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-13)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(covariance))), -1e-11)

    def test_zero_hencky_tensor_reproduces_isotropic_local_geometry(self) -> None:
        normals = interaction.sobol_standard_normals(8, seed=17)
        isotropic = interaction.conditional_line_geometry(1.3, 0.24, normals)
        anisotropic = interaction.conditional_line_geometry(
            1.3, 0.24, normals, aniso=True, H=np.zeros((3, 3))
        )
        for first, second in zip(
            (isotropic.kappa, isotropic.kappa_prime, isotropic.kappa_tau, isotropic.weights),
            (anisotropic.kappa, anisotropic.kappa_prime, anisotropic.kappa_tau, anisotropic.weights),
        ):
            np.testing.assert_allclose(first, second, rtol=2e-12, atol=2e-12)
        with self.assertRaisesRegex(ValueError, "H is required"):
            interaction.conditional_line_geometry(1.3, 0.24, normals, aniso=True)

    def test_anisotropic_curve_jet_uses_exact_line_factor_and_unit_tangent(self) -> None:
        tangent = np.array([[1.0, 0.0, 0.0]])
        r2 = np.array([[0.0, 0.7, 0.0]])
        r3 = np.array([[-0.49, 0.0, 0.14]])
        H = np.diag([2e-6, -0.5e-6, -1.5e-6])
        transformed_tangent, transformed_r2, _, line_factor = (
            interaction.anisotropic_curve_jet(tangent, r2, r3, H)
        )
        np.testing.assert_allclose(np.linalg.norm(transformed_tangent, axis=1), 1.0)
        exact = np.linalg.norm(tangent @ interaction.hencky_stretch_tensor(H).T, axis=1)
        np.testing.assert_allclose(line_factor, exact, rtol=1e-14)
        first_order_line = 1.0 + np.einsum("ni,ij,nj->n", tangent, H, tangent)
        np.testing.assert_allclose(line_factor, first_order_line, rtol=0.0, atol=1e-11)
        exact_kappa = np.linalg.norm(transformed_r2, axis=1)
        normal = r2 / np.linalg.norm(r2, axis=1)[:, None]
        first_order_kappa = np.linalg.norm(r2, axis=1) * (
            1.0 + np.einsum("ni,ij,nj->n", normal, H, normal)
            - 2.0 * np.einsum("ni,ij,nj->n", tangent, H, tangent)
        )
        np.testing.assert_allclose(exact_kappa, first_order_kappa, rtol=0.0, atol=2e-11)

    def test_transformed_trace_is_reparameterized_by_physical_arclength(self) -> None:
        trace = straight_trace()
        H = interaction.uniaxial_hencky_tensor(1.3, axis=0)
        transformed = interaction.transform_contour_trace(trace, H)
        self.assertTrue(transformed.metadata["anisotropic_geometry"])
        for points, tangent in zip(transformed.points, transformed.tangents):
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            np.testing.assert_allclose(distances, transformed.physical_spacing, atol=2e-10)
            np.testing.assert_allclose(np.linalg.norm(tangent, axis=1), 1.0, atol=2e-12)
        zero = interaction.transform_contour_trace(trace, np.zeros((3, 3)))
        np.testing.assert_allclose(zero.points[0], trace.points[0], atol=2e-10)
        config = interaction.NonlocalConfig(0.2, 0.5, 0.1, 0.01, max_points=200)
        table = interaction.nonlocal_change_cell_hamiltonian(
            transformed, interaction.YukawaChangeKernel(0.7, 0.4), config
        )
        np.testing.assert_allclose(table.delta_energy, 0.0, atol=2e-11)

    def test_reduced_conditional_sample_is_deterministic(self) -> None:
        normals1 = interaction.sobol_standard_normals(6, seed=17)
        normals2 = interaction.sobol_standard_normals(6, seed=17)
        np.testing.assert_array_equal(normals1, normals2)
        sample = interaction.conditional_line_geometry(0.8, 0.2, normals1)
        self.assertEqual(len(sample.weights), 64)
        self.assertAlmostEqual(float(np.sum(sample.weights)), 1.0)
        self.assertTrue(np.all(sample.kappa >= 0.0))

    def test_local_fourth_density_and_cell_scaling(self) -> None:
        sample = interaction.LocalGeometrySample(
            np.array([2.0]), np.array([3.0]), np.array([4.0]), np.array([1.0]), 2.0
        )
        expected = (9.0 / 64.0) * 2.0**4 - 3.0**2 / 24.0 - 4.0**2 / 24.0
        self.assertAlmostEqual(float(sample.fourth_order_density[0]), expected)
        default = interaction.local_cell_features(sample, 2.0)
        doubled = interaction.local_cell_features(sample, 2.0, cell_length=1.0)
        np.testing.assert_allclose(doubled, 2.0 * default)

    def test_contour_observables_constant_geometry(self) -> None:
        s = np.linspace(0.0, 5.0, 501)
        kappa = np.full_like(s, 0.4)
        prime = np.full_like(s, 0.2)
        torsion = np.full_like(s, -0.3)
        k2, k4 = interaction.contour_observables(kappa, prime, torsion, s)
        density4 = (9.0 / 64.0) * 0.4**4 - 0.2**2 / 24.0 - 0.3**2 / 24.0
        self.assertAlmostEqual(k2, 5.0 * 0.4**2, places=12)
        self.assertAlmostEqual(k4, 5.0 * density4, places=12)

    def test_torsion_is_recovered_with_zero_curvature_masked(self) -> None:
        sample = interaction.LocalGeometrySample(
            kappa=np.array([0.0, 0.5, 2.0]),
            kappa_prime=np.zeros(3),
            kappa_tau=np.array([0.0, -0.75, 1.0]),
            weights=np.ones(3),
            k_eff=1.0,
        )
        torsion, keep = interaction.torsion_from_local_geometry(
            sample, curvature_floor=1e-12
        )
        np.testing.assert_array_equal(keep, [False, True, True])
        np.testing.assert_allclose(torsion, [-1.5, 0.5])


class NonlocalTests(unittest.TestCase):
    def test_finite_cutoff_factors_and_short_range_limit(self) -> None:
        f2, f4 = interaction.yukawa_cutoff_factors(np.array([1.0, 50.0]))
        self.assertAlmostEqual(f2[0], 1.0 - np.exp(-1.0) * 7.0 / 3.0)
        self.assertAlmostEqual(f4[0], 1.0 - np.exp(-1.0) * 81.0 / 30.0)
        np.testing.assert_allclose([f2[1], f4[1]], 1.0, atol=1e-15)

    def test_finite_cutoff_yukawa_inversion_recovers_synthetic_input(self) -> None:
        expected_g, expected_d, ell = 3.2, 0.35, 1.1
        a, c4 = interaction.yukawa_dimensionless_coefficients(
            expected_g, expected_d, ell
        )
        recovered_g, recovered_d = interaction.solve_yukawa_parameters(a, c4, ell)
        self.assertAlmostEqual(float(recovered_g), expected_g, places=10)
        self.assertAlmostEqual(float(recovered_d), expected_d, places=10)

    def test_relative_mapping_uses_each_conditions_cutoff(self) -> None:
        g0, d0, ell0 = 5.0, 0.4, 1.0
        g1, d1, ell1 = 6.0, 0.3, 0.8
        a0, c40 = interaction.yukawa_dimensionless_coefficients(g0, d0, ell0)
        a1, c41 = interaction.yukawa_dimensionless_coefficients(g1, d1, ell1)
        g_ratio, d_ratio = interaction.relative_yukawa_parameters(
            float(a1-a0), float(c41-c40), float(a0), float(c40),
            ell_kref=ell1, reference_ell_kref=ell0,
        )
        self.assertAlmostEqual(float(g_ratio), g1/g0, places=10)
        self.assertAlmostEqual(float(d_ratio), d1/d0, places=10)

    def test_finite_cutoff_yukawa_inversion_rejects_unphysical_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "no resolved positive Yukawa root"):
            interaction.solve_yukawa_parameters(1.0, 0.5, 1.0)

    def test_relative_reference_normalization_builds_identical_zero_change_kernels(self) -> None:
        reference, target, g_ratio, d_ratio, a0, c40 = interaction.relative_yukawa_potentials(
            2.0, 5.0, 1.0, 0.0, 0.0
        )
        self.assertAlmostEqual(reference.strength, 10.0)
        self.assertAlmostEqual(reference.screening_length, 0.5)
        self.assertEqual(reference, target)
        self.assertAlmostEqual(g_ratio, 1.0)
        self.assertAlmostEqual(d_ratio, 1.0)
        expected_a0, expected_c40 = interaction.yukawa_dimensionless_coefficients(
            5.0, 1.0, 1.0
        )
        self.assertAlmostEqual(float(a0), float(expected_a0))
        self.assertAlmostEqual(float(c40), float(expected_c40))

    def test_conditional_yukawa_family_preserves_normalized_reference(self) -> None:
        family = interaction.conditional_relative_yukawa_family(
            np.array([0.0]), np.array([0.0]),
            np.array([2.0, 5.0]), np.array([0.4, 1.0]), ell_kref=1.0,
        )
        self.assertTrue(np.all(family.valid))
        np.testing.assert_allclose(family.g_over_g0, 1.0)
        np.testing.assert_allclose(family.d_over_d0, 1.0)
        distance = np.array([0.2, 1.0, 3.0])
        profile = interaction.normalized_yukawa_profile(
            distance, family.g_over_g0[:, 0], family.d_over_d0[:, 0]
        )
        expected = np.broadcast_to(np.exp(-distance) / distance, profile.shape)
        np.testing.assert_allclose(profile, expected)

    def test_conditional_yukawa_family_recovers_synthetic_target(self) -> None:
        g0, d0, ell = 4.0, 0.6, 1.0
        g1, d1 = 5.5, 0.45
        a0, c40 = interaction.yukawa_dimensionless_coefficients(g0, d0, ell)
        a1, c41 = interaction.yukawa_dimensionless_coefficients(g1, d1, ell)
        family = interaction.conditional_relative_yukawa_family(
            np.array([float(a1-a0)]), np.array([float(c41-c40)]),
            np.array([g0]), np.array([d0]), ell_kref=ell,
        )
        self.assertTrue(family.valid[0, 0])
        self.assertAlmostEqual(family.g_over_g0[0, 0], g1/g0, places=10)
        self.assertAlmostEqual(family.d_over_d0[0, 0], d1/d0, places=10)

    def test_conditional_yukawa_family_uses_target_cutoff(self) -> None:
        g0, d0, ell0 = 4.0, 0.6, 1.0
        g1, d1, ell1 = 5.5, 0.45, 0.7
        a0, c40 = interaction.yukawa_dimensionless_coefficients(g0, d0, ell0)
        a1, c41 = interaction.yukawa_dimensionless_coefficients(g1, d1, ell1)
        family = interaction.conditional_relative_yukawa_family(
            np.array([float(a1-a0)]), np.array([float(c41-c40)]),
            np.array([g0]), np.array([d0]),
            ell_kref=np.array([ell1]), reference_ell_kref=ell0,
        )
        self.assertTrue(family.valid[0, 0])
        self.assertAlmostEqual(family.g_over_g0[0, 0], g1/g0, places=10)
        self.assertAlmostEqual(family.d_over_d0[0, 0], d1/d0, places=10)

    def test_subtracted_pair_change_vanishes_for_straight_chord(self) -> None:
        separation = np.linspace(0.2, 2.0, 20)
        change = interaction.pair_energy_change(
            separation,
            separation,
            interaction.YukawaPotential(2.0, 0.8),
            interaction.YukawaPotential(1.0, 1.2),
            chord_floor=1e-6,
        )
        np.testing.assert_allclose(change, 0.0, atol=1e-14)

    def test_straight_trace_has_zero_nonlocal_geometry_energy(self) -> None:
        trace = straight_trace()
        config = interaction.NonlocalConfig(0.2, 0.8, 0.1, 0.01, max_points=1000)
        table = interaction.nonlocal_energy_table(
            trace,
            interaction.YukawaPotential(2.0, 0.5),
            interaction.YukawaPotential(1.0, 1.0),
            config,
        )
        np.testing.assert_allclose(table.delta_energy, 0.0, atol=1e-13)
        self.assertEqual(len(np.unique(table.groups)), 2)

    def test_nonlocal_energy_is_integrated_over_the_local_cell(self) -> None:
        trace = straight_trace()
        theta = np.arange(0.0, 3.0, 0.1)
        curved = np.column_stack((np.sin(theta), 1.0 - np.cos(theta), np.zeros_like(theta)))
        trace = interaction.ContourTrace(
            points=(curved, curved + np.array([0.0, 0.0, 1.0])),
            tangents=trace.tangents, r2=trace.r2, r3=trace.r3,
            k_eff=1.0, q_spacing=0.1,
        )
        target = interaction.YukawaPotential(2.0, 0.5)
        reference = interaction.YukawaPotential(1.0, 1.0)
        first = interaction.nonlocal_energy_table(
            trace, target, reference,
            interaction.NonlocalConfig(0.21, 0.8, 0.1, 0.01, max_points=1000),
        )
        second = interaction.nonlocal_energy_table(
            trace, target, reference,
            interaction.NonlocalConfig(0.29, 0.8, 0.1, 0.01, max_points=1000),
        )
        np.testing.assert_allclose(second.delta_energy, first.delta_energy * (0.29 / 0.21))

    def test_absolute_density_and_relative_cell_apis_have_expected_scaling(self) -> None:
        trace = straight_trace()
        theta = np.arange(0.0, 3.0, 0.1)
        curved = np.column_stack((np.sin(theta), 1.0 - np.cos(theta), np.zeros_like(theta)))
        trace = interaction.ContourTrace(
            points=(curved, curved + np.array([0.0, 0.0, 1.0])),
            tangents=trace.tangents, r2=trace.r2, r3=trace.r3,
            k_eff=1.0, q_spacing=0.1,
        )
        potential = interaction.YukawaPotential(2.0, 0.5)
        config = interaction.NonlocalConfig(0.2, 0.8, 0.1, 0.01, max_points=1000)
        absolute = interaction.absolute_nonlocal_energy_density(trace, potential, config)
        relative = interaction.relative_nonlocal_cell_hamiltonian(
            trace, potential, interaction.YukawaPotential(0.0, 0.5), config
        )
        np.testing.assert_allclose(
            relative.delta_energy, config.local_cutoff * absolute.delta_energy
        )

    def test_linked_fourth_order_check_warns_above_tolerance(self) -> None:
        potential = interaction.YukawaPotential(5.0, 1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = interaction.linked_fourth_order_check(
                potential, 1.0, relative_tolerance=0.01
            )
        self.assertFalse(result.adequate)
        self.assertGreater(result.relative_magnitude, result.tolerance)
        self.assertTrue(any("linked K4" in str(item.message) for item in caught))

    def test_relative_sensitivity_dependence_mode_is_default(self) -> None:
        parameter = inspect.signature(
            interaction.fit_relative_nonlocal_correction
        ).parameters["dependence_mode"]
        self.assertEqual(parameter.default, "relative_sensitivity")

    def test_signed_yukawa_change_inversion(self) -> None:
        for expected_g in (3.2, -3.2):
            expected_d, ell = 0.35, 1.1
            delta2, delta4 = interaction.yukawa_change_coefficients(
                expected_g, expected_d, ell
            )
            recovered_g, recovered_d = interaction.solve_yukawa_change(
                delta2, delta4, ell
            )
            self.assertAlmostEqual(float(recovered_g), expected_g, places=10)
            self.assertAlmostEqual(float(recovered_d), expected_d, places=10)

    def test_yukawa_change_rejects_opposite_sign_or_zero_moments(self) -> None:
        with self.assertRaisesRegex(ValueError, "same sign"):
            interaction.solve_yukawa_change(0.1, -0.01, 1.0)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            interaction.solve_yukawa_change(0.0, 0.0, 1.0)

    def test_two_range_change_matches_opposite_sign_moments(self) -> None:
        expected=np.array([-0.12,0.045])
        _,mode,ranges,strengths,condition=interaction.moment_matched_yukawa_change(
            expected[0],expected[1],1.0,mode="auto",two_ranges_kref=(0.5,2.0)
        )
        reconstructed=np.sum(np.asarray([
            interaction.yukawa_change_coefficients(strength,distance,1.0)
            for strength,distance in zip(strengths,ranges)
        ]),axis=0)
        self.assertEqual(mode,"two_range")
        self.assertTrue(np.isfinite(condition))
        np.testing.assert_allclose(reconstructed,expected,rtol=2e-13,atol=2e-13)

    def test_auto_change_falls_back_when_one_range_has_no_finite_cutoff_root(self) -> None:
        expected=np.array([0.1,0.1])
        _,mode,ranges,strengths,_=interaction.moment_matched_yukawa_change(
            expected[0],expected[1],1.0,mode="auto",two_ranges_kref=(0.5,2.0))
        reconstructed=np.sum(np.asarray([
            interaction.yukawa_change_coefficients(strength,distance,1.0)
            for strength,distance in zip(strengths,ranges)
        ]),axis=0)
        self.assertEqual(mode,"two_range")
        np.testing.assert_allclose(reconstructed,expected,rtol=2e-13,atol=2e-13)
        with self.assertRaisesRegex(ValueError,"no resolved positive Yukawa root"):
            interaction.moment_matched_yukawa_change(
                expected[0],expected[1],1.0,mode="one_yukawa")

    def test_direct_change_refit_reinfers_d_delta(self) -> None:
        theta = np.arange(0.0, 8.0, 0.1)
        points = np.column_stack((np.sin(theta), 1.0 - np.cos(theta), np.zeros_like(theta)))
        tangent = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
        r2 = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros_like(theta)))
        r3 = np.column_stack((-np.cos(theta), -np.sin(theta), np.zeros_like(theta)))
        trace = interaction.ContourTrace(
            points=(points, points + np.array([0.0, 0.0, 1.0])),
            tangents=(tangent, tangent.copy()), r2=(r2, r2.copy()),
            r3=(r3, r3.copy()), k_eff=1.0, q_spacing=0.1,
        )
        reference = synthetic_sample(512, 4)
        expected = np.array(interaction.yukawa_change_coefficients(10.0, 0.05, 0.2))
        features = interaction.local_cell_features(reference, 1.0, cell_length=0.2)
        target = interaction.LocalGeometrySample(
            reference.kappa, reference.kappa_prime, reference.kappa_tau,
            reference.weights * np.exp(-(features @ expected)), 1.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = interaction.fit_direct_nonlocal_change_correction(
                trace, reference, target, 1.0, expected[0], expected[1],
                interaction.NonlocalConfig(0.2, 0.8, 0.1, 0.01, max_points=1000),
                cell_length=0.2, maximum_iterations=2,
                coefficient_tolerance=1.0, kernel_relative_tolerance=1.0,
                linked_residual_tolerance=1.0,
            )
        self.assertTrue(result.converged)
        self.assertEqual(result.kernel_history.shape, (2, 2))
        reconstructed = interaction.yukawa_change_coefficients(
            result.g_delta_over_kref, result.d_delta_kref, 0.2
        )
        np.testing.assert_allclose(
            reconstructed,
            [result.corrected_fit.delta_c2_kref, result.corrected_fit.delta_c4_kref3],
            rtol=1e-9, atol=1e-12,
        )

    def test_grouped_dependence_detects_synthetic_signal(self) -> None:
        rng = np.random.default_rng(14)
        groups = np.repeat(np.arange(8), 80)
        state = rng.normal(size=(len(groups), 3))
        energy = 0.8 * np.log1p(state[:, 0] ** 2) + 0.02 * rng.normal(size=len(groups))
        result = interaction.assess_nonlocal_dependence(
            interaction.NonlocalTable(state, energy, groups), folds=4
        )
        self.assertTrue(result.dependent)
        self.assertGreater(result.mean_r2, 0.8)
        self.assertGreater(result.svd_retained_fraction, 0.8)
        self.assertGreater(result.svd_rank, 0)
        model = interaction.fit_conditional_nonlocal_weight(
            interaction.NonlocalTable(state, energy, groups), result, correction_degree=3
        )
        self.assertEqual(model.basis_degree, 3)
        self.assertEqual(len(model.coefficients), 20)

    def test_constant_nonlocal_energy_is_not_dependent(self) -> None:
        rng = np.random.default_rng(22)
        state = rng.normal(size=(160, 3))
        table = interaction.NonlocalTable(state, np.zeros(len(state)), np.repeat(np.arange(8), 20))
        result = interaction.assess_nonlocal_dependence(table, folds=4)
        self.assertFalse(result.dependent)
        self.assertEqual(result.svd_retained_fraction, 0.0)
        np.testing.assert_allclose(result.cv_r2, 0.0)

    def test_independent_conditional_model_is_constant(self) -> None:
        rng = np.random.default_rng(9)
        state = rng.normal(size=(120, 3))
        table = interaction.NonlocalTable(state, rng.normal(scale=0.1, size=120), np.repeat(np.arange(6), 20))
        dependence = interaction.DependenceResult(np.zeros(3), 0.0, 0.0, False, 0.01)
        model = interaction.fit_conditional_nonlocal_weight(table, dependence)
        values = model.beta_delta_free_energy(state[:10])
        np.testing.assert_allclose(values, values[0], atol=1e-14)


class ReweightingTests(unittest.TestCase):
    def test_weighted_correlation_matrix_recovers_linear_dependence(self) -> None:
        x=np.linspace(-2.0,2.0,101)
        values=np.column_stack((x,3*x,-x,np.sin(x)))
        correlation=interaction.weighted_correlation_matrix(
            values,np.linspace(1.0,2.0,len(x)))
        np.testing.assert_allclose(np.diag(correlation),1.0,atol=1e-14)
        self.assertAlmostEqual(correlation[0,1],1.0,places=14)
        self.assertAlmostEqual(correlation[0,2],-1.0,places=14)

    def test_maximum_entropy_recovers_linked_coefficients_and_moments(self) -> None:
        reference = synthetic_sample()
        features = interaction.local_cell_features(reference, 1.0)
        expected = np.array([0.32, -0.18])
        target = interaction.LocalGeometrySample(
            reference.kappa,
            reference.kappa_prime,
            reference.kappa_tau,
            reference.weights * np.exp(-(features @ expected)),
            1.0,
        )
        result = interaction.fit_linked_maximum_entropy(reference, target, 1.0)
        np.testing.assert_allclose(result.coefficients, expected, atol=2e-6)
        np.testing.assert_allclose(result.moment_residual, 0.0, atol=2e-9)
        self.assertGreater(result.reference_effective_fraction, 0.5)

    def test_nonlocal_base_weight_offset_has_the_required_minus_sign(self) -> None:
        reference=synthetic_sample(seed=48)
        features=interaction.local_cell_features(reference,1.0)
        beta_delta_f=0.27*features[:,0]**2-0.11*features[:,1]
        expected=np.array([0.21,-0.08])
        target=interaction.LocalGeometrySample(
            reference.kappa,reference.kappa_prime,reference.kappa_tau,
            reference.weights*np.exp(-beta_delta_f-features@expected),1.0)
        result=interaction.fit_linked_maximum_entropy(
            reference,target,1.0,reference_beta_delta_f_nl=beta_delta_f)
        np.testing.assert_allclose(result.coefficients,expected,atol=3e-6)
        weights=interaction.maximum_entropy_reweighted_weights(
            reference,result,1.0,beta_delta_f_nl=beta_delta_f)
        keep=result.reference_keep
        target_weights=target.weights.copy(); target_weights[~keep]=0
        target_weights/=np.sum(target_weights)
        np.testing.assert_allclose(weights,target_weights,rtol=3e-6,atol=1e-12)

    def test_reduced_nonlocal_corrected_fit_runs_end_to_end(self) -> None:
        reference=synthetic_sample(512,seed=71)
        features=interaction.local_cell_features(reference,1.0,cell_length=0.2)
        expected=np.array([-0.08,0.025])
        target=interaction.LocalGeometrySample(
            reference.kappa,reference.kappa_prime,reference.kappa_tau,
            reference.weights*np.exp(-features@expected),1.0)
        result=interaction.fit_nonlocal_corrected_maximum_entropy(
            straight_trace(),reference,target,1.0,expected[0],expected[1],
            interaction.NonlocalConfig(0.2,0.8,0.1,0.01,max_points=1000),
            cell_length=0.2,maximum_iterations=2,coefficient_tolerance=1e-3)
        self.assertEqual(result.kernel_mode,"two_range")
        self.assertFalse(result.dependence.dependent)
        self.assertFalse(result.shape_dependence_required)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.corrected_fit.coefficients,expected,atol=2e-6)
        self.assertAlmostEqual(float(np.sum(result.corrected_reference_weights)),1.0)

    def test_maximum_entropy_reconstruction_respects_support_and_mask(self) -> None:
        reference = synthetic_sample(seed=17)
        features = interaction.local_cell_features(reference, 1.0)
        target = interaction.LocalGeometrySample(
            reference.kappa,
            reference.kappa_prime,
            reference.kappa_tau,
            reference.weights * np.exp(-(features @ np.array([0.1, 0.03]))),
            1.0,
        )
        train = np.arange(len(reference.weights)) % 2 == 0
        test = ~train
        result = interaction.fit_linked_maximum_entropy(
            reference, target, 1.0, reference_mask=train, target_mask=train
        )
        weights = interaction.maximum_entropy_reweighted_weights(
            reference, result, 1.0, mask=test
        )
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertTrue(np.all(weights[train] == 0.0))
        support = np.all(
            (features >= result.feature_bounds[:, 0])
            & (features <= result.feature_bounds[:, 1]), axis=1
        )
        self.assertTrue(np.all(weights[~support] == 0.0))

    def test_independent_fourth_order_decomposition_recovers_linked_direction(self) -> None:
        linked = np.array([9/64, -1/24, -1/24])
        coefficients = np.concatenate(([0.2], 0.37*linked))
        delta_c4, residual, fraction = interaction.decompose_independent_fourth_coefficients(
            coefficients
        )
        self.assertAlmostEqual(delta_c4, 0.37)
        np.testing.assert_allclose(residual, 0.0, atol=1e-15)
        self.assertLess(fraction, 1e-14)

    def test_local_invariant_observables_reconstruct_J4(self) -> None:
        sample = synthetic_sample(size=64)
        values = interaction.local_invariant_observables(sample, 1.0)
        np.testing.assert_allclose(
            values[:, 4], (9/64)*values[:, 1]-values[:, 2]/24-values[:, 3]/24
        )

    def test_recovers_known_local_cell_coefficients(self) -> None:
        reference = synthetic_sample()
        features = interaction.local_cell_features(reference, 1.0)
        expected = np.array([0.32, -0.18])
        target_weights = reference.weights * np.exp(-(features @ expected))
        target = interaction.LocalGeometrySample(
            reference.kappa,
            reference.kappa_prime,
            reference.kappa_tau,
            target_weights,
            1.0,
        )
        result = interaction.fit_local_cell_reweighting(reference, target, 1.0)
        np.testing.assert_allclose(
            [result.delta_c2_kref, result.delta_c4_kref3], expected, atol=2e-3
        )

    def test_expansion_range_changes_effective_coefficients_inversely(self) -> None:
        reference = synthetic_sample(seed=8)
        features = interaction.local_cell_features(reference, 1.0)
        expected = np.array([0.24, 0.12])
        target = interaction.LocalGeometrySample(
            reference.kappa,
            reference.kappa_prime,
            reference.kappa_tau,
            reference.weights * np.exp(-(features @ expected)),
            1.0,
        )
        short = interaction.fit_local_cell_reweighting(reference, target, 1.0, cell_length=0.5)
        long = interaction.fit_local_cell_reweighting(reference, target, 1.0, cell_length=2.0)
        np.testing.assert_allclose(
            [short.delta_c2_kref, short.delta_c4_kref3], 2.0 * expected, atol=3e-3
        )
        np.testing.assert_allclose(
            [long.delta_c2_kref, long.delta_c4_kref3], 0.5 * expected, atol=3e-3
        )

    def test_support_mask_can_be_applied_to_heldout_samples(self) -> None:
        reference = synthetic_sample(seed=18)
        features = interaction.local_cell_features(reference, 1.0)
        target = interaction.LocalGeometrySample(
            reference.kappa,
            reference.kappa_prime,
            reference.kappa_tau,
            reference.weights * np.exp(-(features @ np.array([0.12, -0.04]))),
            1.0,
        )
        train = np.arange(len(reference.weights)) % 2 == 0
        result = interaction.fit_local_cell_reweighting(
            reference, target, 1.0, reference_mask=train, target_mask=train
        )
        support = interaction.local_cell_support_mask(reference, result, 1.0)
        expected = np.all(
            (features >= result.feature_bounds[:, 0])
            & (features <= result.feature_bounds[:, 1]),
            axis=1,
        )
        np.testing.assert_array_equal(support, expected)
        self.assertTrue(np.any(support & ~train))
        self.assertFalse(np.any(result.reference_keep & ~train))

    def test_flexible_log_ratio_projection_recovers_linked_coefficients(self) -> None:
        reference = synthetic_sample(seed=29)
        expected = np.array([0.21, -0.07])

        class ExactFlexibleFit:
            def log_ratio(self, state, k_ref):
                sample = interaction.LocalGeometrySample(
                    state[:, 0], state[:, 1], state[:, 2],
                    np.ones(len(state)), k_ref,
                )
                return 0.37 - interaction.local_cell_features(sample, k_ref) @ expected

        projection = interaction.project_flexible_fit_to_linked_coefficients(
            reference, ExactFlexibleFit(), 1.0
        )
        np.testing.assert_allclose(
            [projection.delta_c2_kref, projection.delta_c4_kref3],
            expected,
            atol=1e-11,
        )
        self.assertLess(projection.unexplained_fraction, 1e-12)

    def test_projection_standard_errors_shrink_with_sample_count(self) -> None:
        reference = synthetic_sample(size=256, seed=31)

        class InexactFlexibleFit:
            def log_ratio(self, state, k_ref):
                sample = interaction.LocalGeometrySample(
                    state[:, 0], state[:, 1], state[:, 2],
                    np.ones(len(state)), k_ref,
                )
                features = interaction.local_cell_features(sample, k_ref)
                return 0.2 - features @ np.array([0.12, -0.04]) + 0.03 * state[:, 0]**6

        duplicated = interaction.LocalGeometrySample(
            np.tile(reference.kappa, 2),
            np.tile(reference.kappa_prime, 2),
            np.tile(reference.kappa_tau, 2),
            np.ones(2 * len(reference.weights)),
            reference.k_eff,
        )
        first = interaction.project_flexible_fit_to_linked_coefficients(
            reference, InexactFlexibleFit(), 1.0
        )
        second = interaction.project_flexible_fit_to_linked_coefficients(
            duplicated, InexactFlexibleFit(), 1.0
        )
        expected_ratio = (len(reference.weights) - 3) / (2 * len(reference.weights) - 3)
        np.testing.assert_allclose(
            np.diag(second.covariance) / np.diag(first.covariance),
            expected_ratio,
            rtol=1e-10,
        )

    def test_independent_invariants_recover_linked_fourth_order_pattern(self) -> None:
        reference = synthetic_sample(seed=81)
        linked = np.array([0.2, -0.16])
        target = interaction.LocalGeometrySample(
            reference.kappa,
            reference.kappa_prime,
            reference.kappa_tau,
            reference.weights * np.exp(-(interaction.local_cell_features(reference, 1.0) @ linked)),
            1.0,
        )
        result = interaction.fit_independent_invariant_density_ratio(reference, target, 1.0)
        expected = np.array([linked[0], linked[1] * 9.0 / 64.0, -linked[1] / 24.0, -linked[1] / 24.0])
        np.testing.assert_allclose(result.coefficients, expected, atol=3e-3)

    def test_relative_yukawa_identity(self) -> None:
        a0, c40 = interaction.yukawa_dimensionless_coefficients(2.0, 0.4, 1.0)
        g, d = interaction.relative_yukawa_parameters(
            np.zeros(3), np.zeros(3), float(a0), float(c40)
        )
        np.testing.assert_allclose(g, 1.0)
        np.testing.assert_allclose(d, 1.0)
        se_g, se_d = interaction.relative_yukawa_uncertainty(
            0.0, 0.0, np.zeros((2, 2)), float(a0), float(c40)
        )
        self.assertEqual(se_g, 0.0)
        self.assertEqual(se_d, 0.0)

    def test_conditional_short_range_mapping_matches_legacy_algebra(self) -> None:
        delta2 = np.array([0.0, -0.37615443851001806])
        delta4 = np.array([0.0, 0.15189971034204333])
        g_ratio, d_ratio = interaction.short_range_relative_yukawa_parameters(
            delta2, delta4, 5.0 / 8.0, 5.0
        )
        np.testing.assert_allclose(g_ratio[0], 1.0)
        np.testing.assert_allclose(d_ratio[0], 1.0)
        np.testing.assert_allclose(g_ratio[1], 0.1538, atol=5e-4)
        np.testing.assert_allclose(d_ratio[1], 1.608695, atol=1e-6)
        se_g, se_d = interaction.short_range_relative_yukawa_uncertainty(
            0.0, 0.0, np.zeros((2, 2)), 5.0 / 8.0, 5.0
        )
        self.assertEqual(se_g, 0.0)
        self.assertEqual(se_d, 0.0)

    def test_conditional_short_range_family_matches_fixed_mapping(self) -> None:
        delta2 = np.array([0.0, -0.1])
        delta4 = np.array([0.0, 0.2])
        family = interaction.conditional_short_range_yukawa_family(
            delta2, delta4, np.array([5.0]), np.array([1.0])
        )
        expected_g, expected_d = interaction.short_range_relative_yukawa_parameters(
            delta2, delta4, 5.0 / 8.0, 5.0
        )
        self.assertTrue(np.all(family.valid))
        np.testing.assert_allclose(family.g_over_g0[0], expected_g)
        np.testing.assert_allclose(family.d_over_d0[0], expected_d)

    def test_contour_weights_expand_to_normalized_point_distribution(self) -> None:
        observables = np.array([[0.4, -0.1], [0.8, 0.2], [1.2, 0.5]])
        result = interaction.MaximumEntropyResult(
            coefficients=np.array([0.3, -0.2]), covariance=np.eye(2),
            objective=0.0, success=True, message="", target_mass_retained=1.0,
            reference_effective_fraction=1.0,
            feature_bounds=np.array([[-10.0, 10.0], [-10.0, 10.0]]),
            reference_keep=np.ones(3, dtype=bool),
            target_keep=np.ones(3, dtype=bool),
            target_moments=np.zeros(2), fitted_moments=np.zeros(2),
        )
        contour_weights = interaction.contour_maximum_entropy_weights(
            observables, result, 1.0
        )
        expected = np.exp(-(observables @ result.coefficients))
        expected /= expected.sum()
        np.testing.assert_allclose(contour_weights, expected)
        state, point_weights = interaction.trace_point_state_weights(
            straight_trace(), np.array([0.4, 0.6]), 1.0
        )
        self.assertEqual(len(state), len(point_weights))
        self.assertAlmostEqual(float(point_weights.sum()), 1.0)
        blocks = interaction.trace_local_state_blocks(straight_trace(), 0.5)
        self.assertTrue(all(len(block) == 6 for block in blocks))
        self.assertGreater(len(blocks), 2)

    def test_contour_heldout_fit_is_feasible_and_deterministic(self) -> None:
        rng = np.random.default_rng(421)
        reference = np.column_stack((
            np.exp(0.25 * rng.standard_normal(320)),
            0.3 * rng.standard_normal(320),
        ))
        probabilities = np.exp(-(reference @ np.array([0.18, -0.12])))
        probabilities /= probabilities.sum()
        target = reference[rng.choice(len(reference), 360, replace=True,
                                      p=probabilities)]
        first = interaction.fit_contour_heldout_maximum_entropy(
            reference, target, 1.0, support_quantiles=(0.02, 0.98), seed=731
        )
        second = interaction.fit_contour_heldout_maximum_entropy(
            reference, target, 1.0, support_quantiles=(0.02, 0.98), seed=731
        )
        np.testing.assert_array_equal(first.reference_train, second.reference_train)
        np.testing.assert_array_equal(first.target_train, second.target_train)
        np.testing.assert_array_equal(first.target_test_keep, second.target_test_keep)
        np.testing.assert_allclose(first.fit.coefficients, second.fit.coefficients)
        self.assertGreaterEqual(np.count_nonzero(~first.reference_train), 4)
        self.assertGreaterEqual(np.count_nonzero(first.target_test_keep), 4)

    def test_replicate_bootstrap_is_deterministic(self) -> None:
        references = [synthetic_sample(256, seed) for seed in (3, 5, 7)]
        targets = []
        expected = np.array([0.1, -0.05])
        for reference in references:
            features = interaction.local_cell_features(reference, 1.0)
            targets.append(interaction.LocalGeometrySample(
                reference.kappa, reference.kappa_prime, reference.kappa_tau,
                reference.weights * np.exp(-(features @ expected)), 1.0,
            ))
        first = interaction.bootstrap_local_cell_replicates(
            references, targets, 1.0, bootstrap_samples=8, seed=41
        )
        second = interaction.bootstrap_local_cell_replicates(
            references, targets, 1.0, bootstrap_samples=8, seed=41
        )
        np.testing.assert_allclose(first.estimates, second.estimates)
        np.testing.assert_allclose(first.mean, expected, atol=4e-3)


class ArchitectureTests(unittest.TestCase):
    def test_trace_cache_round_trip_and_mismatch(self) -> None:
        trace = straight_trace()
        config = interaction.TraceConfig(grid_size=16, num_modes=16, q_spacing=0.1)
        spectrum = interaction.FitSpectrum(0.0, 1.0, 0.98, 0.2)
        path = HERE / "_cf_interaction_test_trace.npz"
        try:
            interaction.save_trace_cache(path, trace, config, spectrum)
            restored = interaction.load_trace_cache(path, config, spectrum)
            self.assertIsNotNone(restored)
            np.testing.assert_allclose(restored.points[0], trace.points[0])
            mismatch = interaction.load_trace_cache(
                path, interaction.TraceConfig(grid_size=16, num_modes=32, q_spacing=0.1)
            )
            self.assertIsNone(mismatch)
            spectrum_mismatch = interaction.load_trace_cache(
                path, config, interaction.FitSpectrum(0.0, 1.0, 0.97, 0.3)
            )
            self.assertIsNone(spectrum_mismatch)
            transformed = interaction.transform_contour_trace(
                trace, interaction.uniaxial_hencky_tensor(1.2)
            )
            interaction.save_trace_cache(path, transformed, config, spectrum)
            restored_transformed = interaction.load_trace_cache(path, config, spectrum)
            self.assertTrue(restored_transformed.metadata["anisotropic_geometry"])
            np.testing.assert_allclose(
                restored_transformed.metadata["hencky_tensor"],
                transformed.metadata["hencky_tensor"],
            )
        finally:
            if path.exists():
                path.unlink()

    def test_module_import_has_no_filesystem_side_effect(self) -> None:
        before = {path.name for path in HERE.iterdir()}
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(HERE)
        subprocess.run(
            [sys.executable, "-B", "-c", "import cf_ca_interaction"],
            cwd=HERE,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        after = {path.name for path in HERE.iterdir()}
        self.assertEqual(after, before)

    def test_adjacent_comparison_pairs_and_cumulative_changes(self) -> None:
        spectra = [
            interaction.FitSpectrum(salt, 1.0 + salt / 100.0, 1.0, 0.2)
            for salt in (0.0, 10.0, 40.0)
        ]
        pairs = interaction.interaction_comparison_pairs(spectra, "adjacent")
        self.assertEqual(
            [(pair.source_concentration_mM, pair.target_concentration_mM)
             for pair in pairs],
            [(0.0, 10.0), (10.0, 40.0)],
        )
        changes = np.array([[1.0, 2.0], [3.0, 5.0]])
        cumulative = interaction.cumulative_adjacent_coefficient_changes(
            pairs, changes, 10.0
        )
        np.testing.assert_allclose(cumulative[0.0], [-1.0, -2.0])
        np.testing.assert_allclose(cumulative[10.0], [0.0, 0.0])
        np.testing.assert_allclose(cumulative[40.0], [3.0, 5.0])

    def test_single_reference_and_long_baseline_pairs(self) -> None:
        spectra = [
            interaction.FitSpectrum(salt, 1.0, 1.0, 0.2)
            for salt in (0.0, 10.0, 40.0)
        ]
        highest = interaction.interaction_comparison_pairs(
            spectra, "single_reference", "highest"
        )
        self.assertTrue(all(pair.source_concentration_mM == 40.0 for pair in highest))
        baseline = interaction.interaction_comparison_pairs(
            spectra, "long_baseline"
        )
        self.assertEqual(len(baseline), 1)
        self.assertTrue(baseline[0].consistency_check)

    def test_coefficient_scale_and_kernel_accumulation(self) -> None:
        values = interaction.rescale_dimensionless_coefficient_change(
            np.array([2.0, 8.0]), 2.0, 1.0
        )
        np.testing.assert_allclose(values, [1.0, 1.0])
        covariance = interaction.rescale_coefficient_covariance(
            np.eye(2), 2.0, 1.0
        )
        np.testing.assert_allclose(covariance, np.diag([0.25, 1.0 / 64.0]))
        pairs = interaction.interaction_comparison_pairs([
            interaction.FitSpectrum(salt, 1.0, 1.0, 0.2)
            for salt in (0.0, 1.0, 2.0)
        ])
        kernels = [
            interaction.YukawaChangeKernel(1.0, 2.0),
            interaction.YukawaMixtureChangeKernel(
                np.array([2.0, -0.5]), np.array([3.0, 4.0])
            ),
        ]
        cumulative = interaction.cumulative_adjacent_yukawa_changes(
            pairs, kernels, 1.0
        )
        distance = np.array([0.7, 2.1])
        np.testing.assert_allclose(
            cumulative[0.0].evaluate(distance), -kernels[0].evaluate(distance)
        )
        np.testing.assert_allclose(
            cumulative[2.0].evaluate(distance), kernels[1].evaluate(distance)
        )

    def test_sampled_change_profile_is_deterministic_and_finite(self) -> None:
        mean, standard_deviation, retained = interaction.sampled_yukawa_change_profile(
            np.array([[1.0, 0.15], [0.6, 0.08]]),
            np.array([np.diag([1e-4, 1e-6]), np.diag([1e-4, 1e-6])]),
            np.array([1.0, 1.2]), np.array([0.8, 1.5, 3.0]),
            samples=12, seed=71, mode="two_range",
        )
        self.assertEqual(retained, 12)
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(standard_deviation)))
        self.assertTrue(np.all(standard_deviation > 0))

    def test_two_notebook_workflow_and_no_reusable_definitions(self) -> None:
        active = {path.name for path in HERE.glob("cf_ca_interaction*.ipynb")}
        self.assertEqual(active, {
            "cf_ca_interaction_compute.ipynb",
            "cf_ca_interaction_results.ipynb",
        })
        module_source = (HERE / "cf_ca_interaction.py").read_text(encoding="utf-8")
        self.assertIn("-beta_delta_f[keep]", module_source)

        compute_notebook = json.loads(
            (HERE / "cf_ca_interaction_compute.ipynb").read_text(encoding="utf-8")
        )
        compute_source = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in compute_notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compute_tree = ast.parse(compute_source)
        compute_definitions = [
            node for node in ast.walk(compute_tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertEqual(compute_definitions, [])
        self.assertIn("sample_local_geometry_series", compute_source)
        self.assertIn("REFERENCE_CONDITION =", compute_source)
        self.assertIn("select_reference_spectrum", compute_source)
        self.assertIn("highest_salt", compute_source)
        self.assertIn("ANISOTROPIC_GEOMETRY =", compute_source)
        self.assertIn("fitted_hencky_tensors", compute_source)
        self.assertIn("transform_contour_trace", compute_source)
        self.assertIn("anisotropy_inputs.csv", compute_source)
        self.assertIn("fit_flexible_density_ratio", compute_source)
        self.assertIn("RUN_RANDOMWAVE_SAMPLING =", compute_source)
        self.assertIn("RUN_RANDOMWAVE_SAMPLING = True", compute_source)
        self.assertIn("fit_linked_maximum_entropy", compute_source)
        self.assertIn("fit_independent_maximum_entropy", compute_source)
        self.assertIn("local_invariant_moments.csv", compute_source)
        self.assertIn("local_structure_histograms.csv", compute_source)
        self.assertIn("local_structure_correlations.csv", compute_source)
        self.assertIn("weighted_correlation_matrix", compute_source)
        self.assertIn("KS_kappa_prime", compute_source)
        self.assertIn("independent_fourth_order_coefficients.csv", compute_source)
        self.assertIn("RUN_EXPANSION_RANGE_SENSITIVITY =", compute_source)
        self.assertIn("RUN_REPLICATE_UNCERTAINTY =", compute_source)
        self.assertIn("BUILD_REFERENCE_TRACE =", compute_source)
        self.assertIn("build_isotropic_trace", compute_source)
        self.assertIn("maximum_entropy_reweighted_weights", compute_source)
        self.assertIn("RUN_NONLOCAL_DEPENDENCE =", compute_source)
        self.assertIn("absolute_nonlocal_energy_density", compute_source)
        self.assertIn("RUN_NONLOCAL_CORRECTION =", compute_source)
        self.assertIn("fit_nonlocal_corrected_maximum_entropy", compute_source)
        self.assertIn("nonlocal_trace_salts = set(spectrum_by_salt)", compute_source)
        self.assertIn("series_requires_shape_correction = bool(dependent_salts)", compute_source)
        self.assertIn("RUN_NONLOCAL_CORRECTION and series_requires_shape_correction", compute_source)
        self.assertIn("shape_correction_allowed = True", compute_source)
        self.assertIn("RUN_CONTOUR_VALIDATION =", compute_source)
        self.assertIn("RUN_CONTOUR_VALIDATION = False", compute_source)
        self.assertIn("fit_contour_maximum_entropy", compute_source)
        self.assertIn("contour_reconstruction_histograms.csv", compute_source)
        self.assertIn("contour_maximum_entropy_weights", compute_source)
        self.assertIn("target_mean_K2_over_ksource", compute_source)
        self.assertIn("fitted_mean_K4_over_ksource3", compute_source)
        self.assertIn("randomwave_target_K2_over_ksource", compute_source)
        self.assertIn("randomwave_fitted_K4_over_ksource3", compute_source)
        self.assertIn("randomwave_target_k2_density_over_kref2", compute_source)
        self.assertIn("contour_target_k4_density_over_kref4", compute_source)
        self.assertIn('"local_k2": predicted_state', compute_source)
        self.assertIn('"local_k4": (9/64)*predicted_state', compute_source)

        results_notebook = json.loads(
            (HERE / "cf_ca_interaction_results.ipynb").read_text(encoding="utf-8")
        )
        results_source = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in results_notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertNotIn("build_isotropic_trace", results_source)
        self.assertNotIn("sample_local_geometry_series", results_source)
        self.assertIn("conditional_relative_yukawa_family", results_source)
        self.assertIn("measured_local_structure_distributions.png", results_source)
        self.assertIn("measured_local_structure_correlations.png", results_source)
        self.assertIn("local_X_fit.png", results_source)
        self.assertIn("local_k2_k4_fit.png", results_source)
        self.assertIn("local_k2", results_source)
        self.assertIn("local_k4", results_source)
        self.assertIn("randomwave_target_K2_over_ksource", results_source)
        self.assertIn("source_scale**2 / LOCAL_EXPANSION_RANGE_MULTIPLIER", results_source)
        self.assertIn("contour_target_k4_density_over_kref4", results_source)
        self.assertIn("REFERENCE_CONDITION =", results_source)
        self.assertIn("g/g_{\\rm ref}", results_source)
        self.assertIn("ANISOTROPIC_GEOMETRY =", results_source)
        self.assertIn("anisotropy_inputs.csv", results_source)
        self.assertIn("CONDITIONAL_REFERENCE_MODE =", results_source)
        self.assertIn("CONDITIONAL_CUTOFF_MODE =", results_source)
        self.assertIn("short_range_relative_yukawa_parameters", results_source)
        self.assertIn("conditional_{CONDITIONAL_REFERENCE_MODE}_{CONDITIONAL_CUTOFF_MODE}_yukawa_comparison.png", results_source)
        self.assertIn("local_reconstruction_histograms.csv", results_source)
        self.assertIn("P(\\kappa\\tau)", results_source)
        self.assertIn("k2_k4_line_density_comparison.png", results_source)
        self.assertNotIn('moment_change["J4"]', results_source)
        self.assertIn('LOCAL_K4_HISTOGRAM_RANGE = (-0.25, 0.25)', results_source)
        self.assertIn('LOCAL_K4_HISTOGRAM_BINS = 180', compute_source)
        self.assertIn('label="contour trace"', results_source)
        self.assertIn('label="random wave"', results_source)
        self.assertIn("local_reconstruction_model_quality.png", results_source)
        self.assertNotIn("D_i k_", results_source)
        self.assertIn("g_over_g0_common_cutoff_mean", results_source)
        self.assertFalse(any(
            not "".join(cell.get("source", [])).strip()
            for cell in results_notebook["cells"]
        ))
        block_start = "# 1. SETTINGS THAT MUST MATCH BETWEEN COMPUTE AND RESULTS.\n"
        block_end = "# END MATCHED SETTINGS\n"
        compute_matched = compute_source.split(block_start, 1)[1].split(block_end, 1)[0]
        results_matched = results_source.split(block_start, 1)[1].split(block_end, 1)[0]
        self.assertEqual(compute_matched, results_matched)
        self.assertIn("different LOCAL_EXPANSION_RANGE_MULTIPLIER", results_source)


if __name__ == "__main__":
    unittest.main()
