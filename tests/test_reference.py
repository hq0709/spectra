"""spectra.reference: the external reference pair / reference family architecture and pair schedule."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")


def _tiny_unet():
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16,
                                attention_head_dim=4, norm_num_groups=8)


def test_widened_conv_in_starts_as_the_pretrained_model():
    from spectra.reference import widen_conv_in

    u = _tiny_unet().eval()
    x, prior, ctx = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8), torch.randn(2, 1, 16)
    t = torch.tensor([10, 500])
    ref = u(x, t, encoder_hidden_states=ctx).sample
    widen_conv_in(u)
    assert u.conv_in.in_channels == 8 and u.config.in_channels == 8
    out = u(torch.cat([x, prior], 1), t, encoder_hidden_states=ctx).sample
    assert torch.allclose(ref, out, atol=1e-5)


def test_prior_only_ignores_delta_and_prior_delta_does_not():
    from spectra.reference import Conditioner

    po, pd = Conditioner("prior_only", 16, 8), Conditioner("prior_delta", 16, 8)
    h = torch.tensor([24.0, 240.0])
    a = po(h)
    assert a.shape == (2, 1, 16) and torch.equal(a[0], a[1])
    b = pd(h)
    assert b.shape == (2, 1, 16) and not torch.allclose(b[0], b[1])
    with pytest.raises(ValueError):
        Conditioner("prior_delta_actions_typo")


def test_one_pair_schedule_for_all_configurations():
    from spectra.reference import pair_schedule

    pairs = [(i, i + 1, 24.0) for i in range(100)]
    s1, s2 = pair_schedule(pairs, 5, 8, "seed"), pair_schedule(pairs, 5, 8, "seed")
    assert s1 == s2 and len(s1) == 5 and all(len(b) == 8 for b in s1) and max(map(max, s1)) < 100
    assert pair_schedule(pairs, 5, 8, "other") != s1


def test_taix_pairs_window_and_cap():
    from spectra.reference import taix_pairs
    from spectra.types import Film

    films = [Film("p", f"u{d}", f"u{d}", d * 24.0, d, None, None, "AP", "train", "taix") for d in (0, 1, 5, 40, 41)]
    films += [Film(f"q{i}", f"v{i}", f"v{i}", 0.0, 0, None, None, "AP", "train", "taix") for i in range(2)]
    pr = taix_pairs(films, (1, 30), 20, "s")
    assert {(a, b) for a, b, _ in pr} == {("u0", "u1"), ("u0", "u5"), ("u1", "u5"), ("u40", "u41")}
    assert all(24.0 <= h <= 720.0 for _, _, h in pr)
    assert len(taix_pairs(films, (1, 30), 2, "s")) == 2


def test_prior_noise_depends_on_prior_and_seed_only():
    from spectra.reference import prior_noise

    a, b = prior_noise("x", 1009, (1, 4, 2, 2), "cpu"), prior_noise("x", 1009, (1, 4, 2, 2), "cpu")
    assert torch.equal(a, b) and not torch.equal(a, prior_noise("x", 1010, (1, 4, 2, 2), "cpu"))


def test_step_noise_is_a_function_of_seed_and_step():
    from spectra.reference import step_noise

    n1, t1 = step_noise(1, 5, 8, 1000, (4, 2, 2))
    n2, t2 = step_noise(1, 5, 8, 1000, (4, 2, 2))
    assert torch.equal(n1, n2) and torch.equal(t1, t2)
    assert not torch.equal(n1, step_noise(1, 6, 8, 1000, (4, 2, 2))[0])


def test_prior_delta_has_no_unused_parameters():
    from spectra.reference import Conditioner

    c = Conditioner("prior_delta", 16, 8)
    c(torch.tensor([24.0, 48.0])).sum().backward()
    assert all(p.grad is not None for p in c.parameters())


def test_old_8_channel_checkpoints_load_into_the_split_conv_in():
    from spectra.reference import PriorConvIn, upgrade_state_dict, widen_conv_in

    u = _tiny_unet().eval()
    widen_conv_in(u)
    with torch.no_grad():
        u.conv_in.prior.weight.normal_()
    sd = u.state_dict()
    old = {k: v for k, v in sd.items() if not k.startswith("conv_in.")}
    old["conv_in.weight"] = torch.cat([sd["conv_in.noisy.weight"], sd["conv_in.prior.weight"]], 1)
    old["conv_in.bias"] = sd["conv_in.noisy.bias"]
    v = _tiny_unet().eval()
    widen_conv_in(v)
    v.load_state_dict(upgrade_state_dict(old))
    x, t, ctx = torch.randn(2, 8, 8, 8), torch.tensor([10, 500]), torch.randn(2, 1, 16)
    assert isinstance(v.conv_in, PriorConvIn)
    assert torch.allclose(u(x, t, encoder_hidden_states=ctx).sample, v(x, t, encoder_hidden_states=ctx).sample)


def test_split_conv_in_matches_a_widened_8_channel_conv_bitwise():
    from spectra.reference import PriorConvIn

    old = torch.nn.Conv2d(4, 32, 3, padding=1)
    m = PriorConvIn(old)
    with torch.no_grad():
        m.prior.weight.normal_()
    ref = torch.nn.Conv2d(8, 32, 3, padding=1)
    with torch.no_grad():
        ref.weight.copy_(torch.cat([old.weight, m.prior.weight], 1)); ref.bias.copy_(old.bias)
    x = torch.randn(2, 8, 16, 16)
    assert torch.equal(m(x), ref(x))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(m(x), ref(x))


def test_single_group_optimizer_state_migrates_to_two_groups():
    from spectra.reference import Conditioner, upgrade_optimizer_state, widen_conv_in

    u, c = _tiny_unet(), Conditioner("prior_only", 16, 8)
    # legacy layout: one 8-channel conv_in, one AdamW group
    w8 = torch.nn.Conv2d(8, u.conv_in.out_channels, 3, padding=1)
    u.conv_in = w8
    params = list(u.parameters()) + list(c.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()
    v, c2 = _tiny_unet(), Conditioner("prior_only", 16, 8)
    widen_conv_in(v)
    new = upgrade_optimizer_state(sd, v, c2, 1e-3, 1.0)
    pv = [p for n, p in v.named_parameters() if n != "conv_in.prior.weight"] + list(c2.parameters())
    opt2 = torch.optim.AdamW([{"params": pv}, {"params": [v.conv_in.prior.weight]}], lr=1e-3)
    opt2.load_state_dict(new)
    a = opt.state[w8.weight]["exp_avg"]
    assert torch.equal(opt2.state[v.conv_in.noisy.weight]["exp_avg"], a[:, :4])
    assert torch.equal(opt2.state[v.conv_in.prior.weight]["exp_avg_sq"], opt.state[w8.weight]["exp_avg_sq"][:, 4:])
    assert torch.equal(opt2.state[v.conv_in.noisy.bias]["exp_avg"], opt.state[w8.bias]["exp_avg"])
    assert len(opt2.state) == len(opt.state) + 1
