"""Regression tests for ray shapes, masked coordinates, and propagation state."""

import math

import pytest
import torch

from deeplens import GeoLens
from deeplens.geometric_surface import Spheric
from deeplens.light import Ray


def test_squeeze_preserves_feature_axes_and_updates_shape():
    o = torch.ones(1, 2, 3, dtype=torch.float64)
    d = torch.zeros_like(o)
    d[..., 2] = 1.0
    ray = Ray(o, d, wvln=0.55)

    ray.squeeze(0)

    assert ray.shape == (2,)
    assert ray.o.shape == (2, 3)
    assert ray.d.shape == (2, 3)
    assert ray.is_valid.shape == (2,)
    assert ray.stop_dist.shape == (2,)
    for name in ("en", "opl", "bend_penalty"):
        assert getattr(ray, name).shape == (2, 1)
    assert ray.wvln.ndim == 0
    torch.testing.assert_close(ray.centroid(), o.new_ones(3))
    torch.testing.assert_close(ray.rms_error(), o.new_tensor(0.0))


def test_squeeze_of_non_singleton_batch_axis_is_a_noop():
    o = torch.ones(2, 3, 3)
    d = torch.zeros_like(o)
    d[..., 2] = 1.0
    ray = Ray(o, d, wvln=0.55)

    ray.squeeze(0)

    assert ray.shape == (2, 3)
    assert ray.o.shape == (2, 3, 3)


def test_unsqueeze_adds_leading_batch_axis_and_updates_shape():
    o = torch.ones(2, 3, 3)
    d = torch.zeros_like(o)
    d[..., 2] = 1.0
    ray = Ray(o, d, wvln=0.55)

    ray.unsqueeze(0)

    assert ray.shape == (1, 2, 3)
    assert ray.o.shape == (1, 2, 3, 3)
    assert ray.d.shape == (1, 2, 3, 3)
    assert ray.is_valid.shape == (1, 2, 3)
    assert ray.stop_dist.shape == (1, 2, 3)
    for name in ("en", "opl", "bend_penalty"):
        assert getattr(ray, name).shape == (1, 2, 3, 1)
    assert ray.wvln.ndim == 0


@pytest.mark.parametrize("operation", ["squeeze", "unsqueeze"])
@pytest.mark.parametrize("dim", [1, -1])
def test_non_leading_batch_axis_is_rejected_without_modifying_ray(operation, dim):
    d = torch.tensor([[0.0, 0.0, 1.0]]).repeat(1, 2, 1)
    ray = Ray(torch.zeros(1, 2, 3), d, 0.55)
    before = ray.clone()

    with pytest.raises(ValueError, match="expected 0"):
        getattr(ray, operation)(dim)

    for name in ("o", "d", "is_valid", "en", "opl", "bend_penalty", "stop_dist"):
        torch.testing.assert_close(getattr(ray, name), getattr(before, name))
    assert ray.shape == before.shape


@pytest.mark.parametrize("mode", ["geometric", "chief_ray"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_centroid_ignores_invalid_nonfinite_origins_and_gradients(mode, bad):
    o = torch.tensor(
        [[1.0, 2.0, 3.0], [bad, bad, bad]], dtype=torch.float64, requires_grad=True
    )
    ray = Ray(o, o.new_tensor([[0.0, 0.0, 1.0]]).repeat(2, 1), 0.55)
    ray.is_valid[1] = 0
    ray.stop_dist = o.new_tensor([0.5, 0.1])

    centroid = ray.centroid(mode)
    centroid.sum().backward()

    torch.testing.assert_close(centroid, o.new_tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(o.grad, o.new_tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("use_reference", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_rms_ignores_invalid_nonfinite_origins_and_gradients(use_reference, bad):
    o = torch.tensor(
        [[1.0, 2.0, 0.0], [3.0, 2.0, 0.0], [bad, bad, bad]],
        dtype=torch.float64,
        requires_grad=True,
    )
    ray = Ray(o, o.new_tensor([[0.0, 0.0, 1.0]]).repeat(3, 1), 0.55)
    ray.is_valid[2] = 0
    reference = o.new_tensor([0.0, 2.0, 0.0]) if use_reference else None

    rms = ray.rms_error(reference)
    rms.backward()

    expected = 5.0**0.5 if use_reference else 1.0
    assert rms.item() == pytest.approx(expected, abs=2e-6)
    assert torch.isfinite(o.grad).all()
    torch.testing.assert_close(o.grad[2], o.new_zeros(3))


def test_all_invalid_nonfinite_bundle_has_infinite_rms_and_zero_gradients():
    o = torch.full((2, 3), float("nan"), requires_grad=True)
    ray = Ray(o, torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1), 0.55)
    ray.is_valid.zero_()

    rms = ray.rms_error()
    rms.backward()

    torch.testing.assert_close(ray.centroid(), o.new_zeros(3))
    assert torch.isposinf(rms)
    torch.testing.assert_close(o.grad, o.new_zeros(2, 3))


def test_prop_to_rejects_float32_coherent_rays_before_mutation():
    ray = Ray(
        torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 1.0]]), 0.55, is_coherent=True
    )
    before = ray.clone()

    with pytest.raises(ValueError, match="float64"):
        ray.prop_to(10.0)

    torch.testing.assert_close(ray.o, before.o)
    torch.testing.assert_close(ray.opl, before.opl)


def test_prop_to_invalid_nonfinite_origin_does_not_poison_direction_gradients():
    o = torch.tensor([[0.0, 0.0, 0.0], [float("nan")] * 3], requires_grad=True)
    d = torch.tensor([[0.0, 0.0, 1.0]] * 2, requires_grad=True)
    ray = Ray(o, d, 0.55)
    ray.is_valid[1] = 0

    ray.prop_to(10.0)
    ray.o[0].sum().backward()

    torch.testing.assert_close(o.grad, o.new_tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(
        d.grad, d.new_tensor([[10.0, 10.0, 0.0], [0.0, 0.0, 0.0]])
    )


def test_prop_to_preserves_negative_near_parallel_direction():
    ray = Ray(
        torch.zeros(1, 3, dtype=torch.float64),
        torch.tensor([[1.0, 0.0, -1e-13]], dtype=torch.float64),
        0.55,
    )

    ray.prop_to(-1.0)

    assert torch.isfinite(ray.o).all()
    assert ray.o[0, 0] > 0
    assert ray.o[0, 2] < 0


def test_prop_to_supports_per_ray_depth_and_refractive_index():
    o = torch.zeros(2, 3, dtype=torch.float64)
    ray = Ray(o, o.new_tensor([[0.0, 0.0, 1.0]]).repeat(2, 1), 0.55, is_coherent=True)

    ray.prop_to(o.new_tensor([10.0, 20.0]), n=o.new_tensor([1.0, 1.5]))

    torch.testing.assert_close(
        ray.o, o.new_tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]])
    )
    torch.testing.assert_close(ray.opl, o.new_tensor([[10.0], [30.0]]))


def test_trace_reanchors_far_float32_bundle_before_intersecting(device_auto):
    """`trace` re-anchors far float32 rays so the shallow sag survives.

    A float32 origin at z=-1e6 cancels against the intersection distance and
    loses the low-order sag; `GeoLens.trace` moves such bundles to z=-10 mm
    first. `Spheric.intersect` no longer re-anchors on its own, so this is the
    only place the guarantee lives.
    """
    lens = GeoLens(device=device_auto)
    lens.surfaces = [
        Spheric(c=1e-4, r=40.0, d_next=0.0, mat2="air", device=device_auto)
    ]
    lens.d_sensor = torch.tensor(0.0, device=device_auto)

    o = torch.tensor([[3.0, 4.0, -1e6]], dtype=torch.float32, device=device_auto)
    ray = Ray(o, o.new_tensor([[0.0, 0.0, 1.0]]), 0.55, device=device_auto)
    traced, _ = lens.trace(ray)

    assert traced.is_valid.all()
    assert traced.o[0, 2].item() == pytest.approx(0.00125, abs=5e-7)


def _far_oblique_bundle(z_obj, fov_deg, num_rays=9, pupil_r=8.0, pupil_z=20.0):
    """A fan from one far off-axis object point, aimed across the pupil.

    Built in float64 so the caller can cast a copy to float32 and attribute any
    difference to the trace rather than to how the bundle was sampled.
    """
    oy = math.tan(math.radians(fov_deg)) * abs(z_obj)
    o = torch.tensor([[0.0, oy, z_obj]], dtype=torch.float64).repeat(num_rays, 1)
    ty = torch.linspace(-pupil_r, pupil_r, num_rays, dtype=torch.float64)
    target = torch.stack(
        [
            torch.zeros(num_rays, dtype=torch.float64),
            ty,
            torch.full((num_rays,), pupil_z),
        ],
        dim=-1,
    )
    return o, target - o


@pytest.mark.parametrize("fov_deg", [10.0, 20.0])
def test_trace_reanchors_far_oblique_float32_bundle(fov_deg, device_auto):
    """The forward re-anchor must hold for *oblique* far bundles, not just axial.

    `test_trace_reanchors_far_float32_bundle_before_intersecting` traces
    `d = [0, 0, 1]`, where the transverse coordinate is identically zero and the
    lateral cancellation cannot show. An off-axis object point at the library
    default `DEPTH = -20000` has `o_y ≈ 3.5e3`, which cancels against the
    intersection distance exactly the way `o_z` does. Without the re-anchor this
    bundle lands ~70 µm (≈5 sensor pixels on a typical lens) off; with it, the
    float32 trace tracks the float64 reference to well under a micron.
    """
    lens = GeoLens("./datasets/lenses/camera/ef50mm_f1.8.json", device=device_auto)
    lens.astype(torch.float64)

    o64, d64 = _far_oblique_bundle(-20000.0, fov_deg)
    ref = lens.trace2sensor(Ray(o64.clone(), d64.clone(), 0.587, device=device_auto))
    got = lens.trace2sensor(
        Ray(o64.to(torch.float32), d64.to(torch.float32), 0.587, device=device_auto)
    )

    both = (ref.is_valid > 0) & (got.is_valid > 0)
    assert both.sum() > 0, "reference bundle fully vignetted; fixture is wrong"
    err = (got.o[both][:, :2].double() - ref.o[both][:, :2]).abs().max()
    assert err < 1e-3, f"float32 oblique trace drifted {float(err) * 1e3:.3f} um"


def test_backward_trace_reanchors_far_float32_bundle(device_auto):
    """`backward_tracing` needs the same guard as `forward_tracing`.

    The re-anchor originally lived in `Spheric.intersect` and fired on
    `|z| > 100` in both signs; moving it into `forward_tracing` left the
    backward path unguarded. A caller-supplied bundle entering from far +z was
    then wrong by millimetres and lost rays outright. `sample_sensor` origins
    sit at local z=0, so the library's own reverse-rendering path never
    exercised this.
    """
    lens = GeoLens("./datasets/lenses/camera/ef50mm_f1.8.json", device=device_auto)
    lens.astype(torch.float64)

    o64, d64 = _far_oblique_bundle(1e6, 5.0, pupil_r=6.0, pupil_z=30.0)
    assert bool((d64[..., 2] < 0).all()), "fixture must produce backward rays"

    ref, _ = lens.trace(Ray(o64.clone(), d64.clone(), 0.587, device=device_auto))
    got, _ = lens.trace(
        Ray(o64.to(torch.float32), d64.to(torch.float32), 0.587, device=device_auto)
    )

    both = (ref.is_valid > 0) & (got.is_valid > 0)
    assert both.sum() >= 6, (
        f"only {int(both.sum())} rays survived the backward trace; without the "
        "re-anchor the float32 bundle loses most of them"
    )
    err = (got.o[both][:, :3].double() - ref.o[both][:, :3]).abs().max()
    assert err < 1e-2, f"float32 backward trace drifted {float(err) * 1e3:.3f} um"
