"""Acceptance gate #1 of E13-01 -- architecture guarantees Truth cannot
reach guidance -- enforced three ways, all from the outside:

1. the feed refuses every record type that is not an Estimate (TypeError
   naming what it was handed);
2. the guidance module :mod:`orbitopt.gnc.feed` has no *name* for truth at
   all (AST scan): a leak needs something to leak through, and the module
   that defines the guidance side of the boundary does not even contain the
   word as an identifier;
3. what guidance DOES receive (an Estimate) is checked by the harness on the
   way to the run log (a command whose source is not a logged, causally
   readable estimate is refused) -- tested in test_e13_01_replay.py."""
import ast
import pathlib

import pytest

from orbitopt.gnc import (Command, Estimate, GNCContractError, GuidanceInput, GuidanceLaw,
                          Observation, ControlEvent, TruthSample, guidance_input_from,
                          run_guidance_step)

import test_e13_01_gnc_contracts as tc


class ConstantLaw:
    """A law that would be cheating: returns a fixed burn from ANY input."""

    def delta_v(self, estimate):
        assert isinstance(estimate, Estimate)
        return (1e-3, 0.0, -1e-3)


def test_push_refuses_plant_state_measurements_and_commands():
    feed = GuidanceInput()
    for wrong in (tc.truth(), tc.observation(), tc.command(), "state vector", 3.0):
        with pytest.raises(TypeError, match="Estimate records only"):
            feed.push(wrong)
    assert len(feed) == 0


def test_push_accepts_estimates_and_orders_them():
    feed = GuidanceInput()
    feed.push(tc.estimate(rec="est-a", t_update=2.0))
    feed.push(tc.estimate(rec="est-b", t_update=4.0))
    assert feed.latest("chaser", "tgt").record_id == "est-b"
    with pytest.raises(GNCContractError, match="older"):
        feed.push(tc.estimate(rec="est-c", t_update=3.0))  # rewind refused


def test_no_estimate_yet_is_an_error_not_a_fallback():
    feed = GuidanceInput()
    with pytest.raises(GNCContractError, match="has not updated"):
        feed.latest("chaser")


def test_multi_target_requires_naming_target():
    feed = GuidanceInput()
    for cid, tid in (("chaser", "tgtA"),):
        est = tc.estimate(rec="estA")
        object.__setattr__(est.relative_state, "target_id", "tgtA")
        feed.push(est)
    est_b = tc.estimate(rec="estB")
    object.__setattr__(est_b.relative_state, "target_id", "tgtB")
    feed.push(est_b)
    with pytest.raises(GNCContractError, match="several targets"):
        feed.latest("chaser")
    assert feed.latest("chaser", "tgtB").record_id == "estB"


def test_feed_has_no_path_to_plant_state():
    # slots are the whole object; the only attribute is the estimate map
    assert GuidanceInput.__slots__ == ("_latest",)
    feed = GuidanceInput()
    assert [a for a in dir(feed) if "truth" in a.lower() or "plant" in a.lower()] == []


def _identifiers(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node.name)
        elif isinstance(node, ast.arg):
            out.append(node.arg)
    return out


def test_guidance_module_cannot_even_name_truth():
    src = pathlib.Path(__file__).resolve().parent.parent / "src/orbitopt/gnc/feed.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    offenders = [n for n in _identifiers(tree) if "truth" in n.lower() or "plant" in n.lower()]
    assert offenders == [], f"guidance-side module names plant state: {offenders}"


def test_law_signature_is_the_whole_worldview():
    # GuidanceLaw.delta_v(self, estimate) -- one data argument, no plant handle
    import inspect
    sig = inspect.signature(GuidanceLaw.delta_v)
    assert list(sig.parameters) == ["self", "estimate"]


def test_run_guidance_step_returns_estimate_and_vector():
    feed = guidance_input_from([tc.estimate()])
    est, dv = run_guidance_step(ConstantLaw(), feed, "chaser", "tgt")
    assert est.record_id == "est-00001"
    assert dv == pytest.approx((1e-3, 0.0, -1e-3))


def test_run_guidance_step_refuses_broken_laws():
    feed = guidance_input_from([tc.estimate()])

    class BadLaw:
        def delta_v(self, estimate):
            return 7.0  # not a vector

    with pytest.raises(TypeError, match="3-vector"):
        run_guidance_step(BadLaw(), feed, "chaser")

    class NaNLaw:
        def delta_v(self, estimate):
            return (float("nan"), 0.0, 0.0)

    with pytest.raises(TypeError, match="finite"):
        run_guidance_step(NaNLaw(), feed, "chaser")
