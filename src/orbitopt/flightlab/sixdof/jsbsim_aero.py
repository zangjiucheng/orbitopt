"""E1-04 (task #253): production JSBSim aerodynamics for the 6-DOF path.

The E4 spike airframe (FL-401 #206, formalised by F-01 #226) deliberately
flies with *identically zero* aero -- it was a TVC / propellant / mass paper
test, and the ``jsbsim`` row of docs/flightlab.md said so.  This module lifts
that boundary for the **production** vehicle: it compiles a Flight Lab aero
document (the same :class:`~orbitopt.flightlab.sixdof.aero.AeroModel` the
native rigid-body plant flies, ``data/flightlab/aero/*.yaml``) into a JSBSim
aircraft XML whose ``<aerodynamics>`` section *is* that coefficient model, so
one ascent scenario can be flown by both 6-DOF engines off one set of tables.

Why a generator rather than a hand-written XML
    "Same aero tables as native" has to be a checked claim, not a promise in a
    comment.  The production aircraft is therefore a derived artefact: the aero
    document is the source, :func:`regenerate` (also
    ``python -m orbitopt.flightlab.sixdof.jsbsim_aero``) rewrites the packaged
    XML from it, and the test suite asserts the checked-in file is identical to
    a fresh generation.  Edit the YAML, regenerate, both sides move together.

From body-axis coefficients to JSBSim axes
    ``AeroModel`` tables are body-axis (``Fx = -qS ca``, ``Fz = -qS cn``,
    ``Fy = +qS cy``, couple ``qS (b cl, c cm, c cn_yaw)`` about the CG).
    JSBSim's ``DRAG``/``LIFT`` axes are stability axes (along and across the
    air-relative velocity); ``SIDE``/``ROLL``/``PITCH``/``YAW`` are body axes
    already.  The conversion is the textbook rotation through alpha, applied at
    the table nodes::

        CD = ca cos(a) + cn sin(a)        CL = cn cos(a) - ca sin(a)

    so the force vector JSBSim applies is the body-axis model's -- exactly at
    the nodes, and within the tables' own linear-interpolation error in
    between, which :func:`promotion_fidelity` measures.  The parts that do not
    rotate through alpha are taken from the sideslip plane: the beta-induced
    drag increment ``CD_beta = ca(M,0,b) - ca(M,0,0)``, ``CY(M,b)`` and
    ``CN_yaw(M,b)``.  An aero document that varies in a way the generated terms
    cannot express -- ``cy`` with alpha, ``cm`` with beta, non-zero ``cn`` at
    zero alpha with sideslip, non-zero roll coefficient -- is *rejected* by
    :func:`aerodynamics_block` with a message telling the author to write the
    ``<aerodynamics>`` section by hand.  Silently dropping a term is the failure
    mode this module must not have.

Rate damping and the reference point
    ``cl_p`` / ``cm_q`` / ``cn_yaw_r`` (the linear stand-ins the aero module
    documents) become JSBSim's usual damping product with ``aero/{bi,ci}2vel``
    and the body rate.  The aerodynamic reference point is placed at the
    airframe's *dry* CG: the aero module's couple is CG-referenced while JSBSim
    applies its forces at a fixed ``AERORP``, so a travelling CG shows up as a
    force-times-CG-shift couple. The generated correction cancels that extra
    couple at the live CG; the remaining interpolation and engine differences
    are quantified in the E1-04 ADR and by
    :mod:`orbitopt.flightlab.sixdof.production_compare`.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Sequence

from orbitopt.flightlab.sixdof.aero import AeroError, AeroModel, load_packaged_aero

__all__ = [
    "AeroError",
    "BASE_AIRCRAFT",
    "PRODUCTION_AERO_MODEL",
    "PRODUCTION_AIRCRAFT",
    "ProductionAircraft",
    "PromotionFidelity",
    "aerodynamics_block",
    "aircraft_aero_note",
    "base_aircraft_xml",
    "main",
    "packaged_production_aircraft",
    "production_aircraft_xml",
    "production_model",
    "promotion_fidelity",
    "regenerate",
]

#: The aero document the production JSBSim vehicle is compiled from, the
#: airframe it inherits masses / propulsion / FCS from (the parity source's
#: ``jsbsim_aircraft``), and the aircraft name it is written under.
PRODUCTION_AERO_MODEL = "spike_stage"
BASE_AIRCRAFT = "spike_stage"
PRODUCTION_AIRCRAFT = "spike_stage_aero"

#: Coefficient differences below this are treated as "the document does not
#: actually vary along that axis": node values are floats produced by the same
#: arithmetic, so anything at this level is rounding, not aerodynamics.
SEPARABILITY_TOL = 1e-9

_DEG = math.pi / 180.0


@dataclass(frozen=True)
class ProductionAircraft:
    """One generated production aircraft: the XML text, the name it loads under,
    the aero document it came from, and where the aerodynamic reference point
    sits (structural x [m] aft of the nose)."""

    name: str
    xml: str
    aero_model: str
    cg_x_m: float


@dataclass(frozen=True)
class PromotionFidelity:
    """How far the generated JSBSim terms are from the body-axis model they were
    compiled from, measured on a dense grid inside the tabulated envelope.

    Each entry is the largest ``|generated - model|`` over the grid, for the
    coefficient as *the native model reports it* (the generated stability-axis
    values are rotated back into body axes first).  Non-zero by construction:
    JSBSim linearly interpolates the rotated tables in ``(Mach, alpha)``, the
    native model interpolates the body tables in ``(Mach, alpha, beta)`` and
    rotates afterwards, and rotating through ``cos/sin`` does not commute with
    interpolating.  That residual is the honest price of the promotion, and the
    E1-04 test asserts it stays a rounding error next to the coefficients."""

    ca: float
    cn: float
    cy: float
    cm: float
    cn_yaw: float
    #: The grid the numbers came from, for the report.
    samples: int

    def worst(self) -> float:
        return max(self.ca, self.cn, self.cy, self.cm, self.cn_yaw)


# --------------------------------------------------------------------------
# reading + validating the document
# --------------------------------------------------------------------------

def _nodes(model: AeroModel, key: str, axis: str) -> tuple[float, ...]:
    table = model.tables[key]
    if axis not in table.axes:
        raise AeroError(f"{key}: the table is not tabulated on {axis}")
    return tuple(float(v) for v in table.axis_values[axis])


def _at(model: AeroModel, key: str, mach: float, alpha_deg: float, beta_deg: float) -> float:
    return float(model.tables[key].evaluate(mach=mach, alpha_deg=alpha_deg,
                                            beta_deg=beta_deg))


def _everywhere(model: AeroModel, key: str):
    """Yield ``(mach, alpha_deg, beta_deg)`` over every node of a table."""
    table = model.tables[key]
    for mach in _nodes(model, key, "mach"):
        if "alpha_deg" in table.axes:
            alphas = _nodes(model, key, "alpha_deg")
        else:
            alphas = (0.0,)
        if "beta_deg" in table.axes:
            betas = _nodes(model, key, "beta_deg")
        else:
            betas = (0.0,)
        for alpha in alphas:
            for beta in betas:
                yield mach, alpha, beta


def _max_spread(model: AeroModel, key: str, grouped: str, other: str) -> float:
    """Largest within-line spread of a table across ``other``, grouping the
    remaining axis by ``grouped`` -- i.e. how much the coefficient varies along
    ``other`` once everything else is pinned."""
    table = model.tables[key]
    if other not in table.axes:
        return 0.0
    if grouped not in table.axes:
        raise AeroError(f"{key}: cannot check {other}-independence without a {grouped} axis")
    worst = 0.0
    for mach in _nodes(model, key, "mach"):
        for g in _nodes(model, key, grouped):
            coords = {"mach": mach, grouped: g}
            line = [_at(model, key, **coords, **{other: v})
                    for v in _nodes(model, key, other)]
            worst = max(worst, max(line) - min(line))
    return worst


def _check_expressible(model: AeroModel) -> None:
    """Reject an aero document the generated ``<aerodynamics>`` block cannot
    reproduce.  Each rule names the coefficient and the axis whose variation is
    lost, so the fix is either a hand-written block or a different model -- not
    a silent omission."""
    problems: list[str] = []

    roll = max((abs(_at(model, "cl", m, a, b)) for m, a, b in _everywhere(model, "cl")),
               default=0.0)
    if roll > SEPARABILITY_TOL:
        problems.append(f"cl is non-zero (max |cl| = {roll:g}): the generated ROLL axis "
                        "carries only roll-rate damping; write the block by hand for "
                        "a lifting body")
    for key, axis, grouped in (("cy", "alpha_deg", "beta_deg"),
                               ("cm", "beta_deg", "alpha_deg"),
                               ("cn", "beta_deg", "alpha_deg"),
                               ("cn_yaw", "alpha_deg", "beta_deg")):
        spread = _max_spread(model, key, grouped, axis)
        if spread > SEPARABILITY_TOL:
            problems.append(f"{key} varies along {axis} (spread {spread:g}): the generated "
                            f"{key.upper()} term is tabulated on {grouped} only")
    # Drag's sideslip dependence must be additive in alpha and beta: the block
    # carries one CD_beta(M, beta) increment, so a document whose beta drag
    # increment changes with alpha cannot be expressed.
    if "beta_deg" in model.tables["ca"].axes:
        if 0.0 not in _nodes(model, "ca", "alpha_deg") or 0.0 not in _nodes(model, "ca", "beta_deg"):
            problems.append("ca tabulates beta without a zero alpha or beta node: the CD_beta "
                            "increment is defined against the (alpha=0, beta=0) node")
        else:
            worst = 0.0
            for mach, alpha, beta in _everywhere(model, "ca"):
                cross = (_at(model, "ca", mach, alpha, beta)
                         - _at(model, "ca", mach, alpha, 0.0)
                         - _at(model, "ca", mach, 0.0, beta)
                         + _at(model, "ca", mach, 0.0, 0.0))
                worst = max(worst, abs(cross))
            if worst > SEPARABILITY_TOL:
                problems.append(f"ca is not additive in (alpha, beta) (cross term {worst:g}): "
                                "the generated CD_beta(M, beta) is alpha-independent")

    if problems:
        raise AeroError(f"{model.name}: cannot be promoted to a JSBSim aircraft:\n  "
                        + "\n  ".join(problems))


# --------------------------------------------------------------------------
# emitting the <aerodynamics> section
# --------------------------------------------------------------------------

def _num(x: float) -> str:
    return repr(float(x))


def _table(name: str, row_axis: str, model: AeroModel, key: str, cell) -> list[str]:
    """One JSBSim 2-D table: alpha or sideslip down the rows, Mach along them.

    ``cell(mach, row_value)`` is the coefficient the native model reports at
    that node, so the table holds the document's own numbers in whatever axis
    the caller asked for."""
    rows = _nodes(model, key, row_axis)
    cols = _nodes(model, key, "mach")
    out = [f'<table name="{name}" internal="1">',
           f'  <independent_var>{row_axis}</independent_var>',
           '  <independent_var>Mach</independent_var>',
           '  <tableData extension="linear">']
    out.append('    ' + ' '.join(_num(c) for c in cols))
    for r in rows:
        out.append('    ' + ' '.join(_num(cell(m, r)) for m in cols))
    out += ['  </tableData>', '</table>']
    return out


def _end(tag: str) -> str:
    # Built at runtime: a literal close-tag for the function element gets
    # mangled by the tooling that shuttles this file around.
    return '<' + '/' + tag + '>'


def _product(name, description, properties, table, value):
    # A coefficient function: qbar * S * [length] * (table or value).
    out = [f'<function name="{name}">',
           f'  <description>{description}</description>',
           '  <product>',
           '    <property>aero/qbar-psf</property>',
           '    <property>metrics/Sw-sqft</property>']
    out += [f'    <property>{p}</property>' for p in properties]
    if table is not None:
        out += ['    ' + line for line in table]
    if value is not None:
        out.append(f'    <value>{_num(value)}</value>')
    out += ['  </product>', _end('function')]
    return out




IN_PER_M = 39.37007874015748
M_TO_FT = 1.0 / 12.0


def _matrix_table(row_var: str, row_nodes: Sequence[float],
                  col_nodes: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[str]:
    # One JSBSim 2-D table.  The row values are written in the unit the
    # independent variable is named for (radians for alpha/sideslip), the
    # columns are Mach; row 0 is the header line of JSBSim's tableData.
    # JSBSim's own element spelling (see the bundled Shuttle.xml): the row
    # variable looks up rows, the column variable the abscissa.
    out = ['<table>',
           f'  <independentVar lookup="row">{row_var}</independentVar>',
           '  <independentVar lookup="column">velocities/mach</independentVar>',
           '  <tableData extension="linear">']
    out.append('    ' + ' '.join(_num(c) for c in col_nodes))
    for row, r in zip(matrix, row_nodes):
        out.append('    ' + _num(r) + ' ' + ' '.join(_num(v) for v in row))
    out += ['  ' + _end('tableData'), _end('table')]
    return out


def _rot_pair(model: AeroModel, mach: float, alpha_deg: float) -> tuple[float, float]:
    """(CD, CL) at a (Mach, alpha) point of the zero-sideslip plane: the
    body-axis ca/cn the document reports, rotated into stability axes."""
    ca = _at(model, 'ca', mach, alpha_deg, 0.0)
    cn = _at(model, 'cn', mach, alpha_deg, 0.0)
    c, s = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))
    return (ca * c + cn * s, cn * c - ca * s)


def _cd_beta(model: AeroModel, mach: float, beta_deg: float) -> float:
    """The sideslip drag increment: ca at (M, 0, beta) minus ca at (M, 0, 0).

    Only meaningful because :func:`_check_expressible` established the
    increment is additive in beta, i.e. independent of alpha."""
    return (_at(model, 'ca', mach, 0.0, beta_deg)
            - _at(model, 'ca', mach, 0.0, 0.0))




def _indented(lines: Sequence[str], pad: int = 2) -> list[str]:
    space = ' ' * pad
    return [space + line for line in lines]


def _axis(name: str, children: Sequence[Sequence[str]]) -> list[str]:
    lines = [line for child in children for line in child]
    if not lines:
        return [f'<axis name="{name}"></axis>']
    return ([f'<axis name="{name}">'] + _indented(lines)
            + ['  ' + _end('axis')])


def _zero(name: str, description: str) -> list[str]:
    # An explicit zero, for an axis the document says nothing about.
    return _product('aero/coefficient/' + name, description, [], None, 0.0)


def _nodes_alphas(model: AeroModel, key: str) -> list[float]:
    table = model.tables[key]
    return list(_nodes(model, key, 'alpha_deg')) if 'alpha_deg' in table.axes else [0.0]


def _nodes_betas(model: AeroModel, key: str) -> list[float]:
    table = model.tables[key]
    return list(_nodes(model, key, 'beta_deg')) if 'beta_deg' in table.axes else [0.0]


def aerodynamics_block(model: AeroModel, *, cg_x_m: float | None = None) -> str:
    """The JSBSim aerodynamics section for one aero document, as XML text.

    With ``cg_x_m`` (the AERORP this block will be installed under, at the
    dry CG) the moment terms also carry the CG-coupling cancellation of
    :func:`_cg_cancel`, so the aircraft reproduces the document's
    CG-referenced couple at every propellant load.

    Raises AeroError if the document varies in a way the generated terms
    cannot express: a coefficient silently dropped is worse than one refused.
    """
    _check_expressible(model)
    machs = _nodes(model, 'ca', 'mach')
    drag: list[str] = []
    lift: list[str] = []
    side: list[str] = []
    pitch: list[str] = []
    yaw: list[str] = []
    roll: list[str] = []

    alphas = _nodes_alphas(model, 'ca')
    alpha_rad = [math.radians(a) for a in alphas]
    drag.append(_product('aero/coefficient/CD0',
                         'axial force (body -x): ca/cn rotated into stability axes',
                         [], _matrix_table('aero/alpha-rad', alpha_rad, machs,
                                           [[_rot_pair(model, m, a)[0] for m in machs]
                                            for a in alphas]), None))
    lift.append(_product('aero/coefficient/CL0',
                         'normal force (body -z): ca/cn rotated into stability axes',
                         [], _matrix_table('aero/alpha-rad', alpha_rad, machs,
                                           [[_rot_pair(model, m, a)[1] for m in machs]
                                            for a in alphas]), None))

    if 'beta_deg' in model.tables['ca'].axes:
        betas = _nodes_betas(model, 'ca')
        drag.append(_product('aero/coefficient/CDbeta',
                             'sideslip drag increment over the zero-sideslip plane',
                             [], _matrix_table('aero/beta-rad',
                                               [math.radians(b) for b in betas], machs,
                                               [[_cd_beta(model, m, b) for m in machs]
                                                for b in betas]), None))
    if 'beta_deg' in model.tables['cy'].axes:
        betas = _nodes_betas(model, 'cy')
        side.append(_product('aero/coefficient/CYbeta', 'side force (body +y)', [],
                             _matrix_table('aero/beta-rad',
                                           [math.radians(b) for b in betas], machs,
                                           [[_at(model, 'cy', m, 0.0, b) for m in machs]
                                            for b in betas]), None))

    if cg_x_m is not None:
        pitch.append(_cg_cancel('CMcg', 'CG coupling of the normal force, cancelled '
                                '(the document couples about the live CG)',
                                'aero/alpha-rad', alpha_rad, machs,
                                [[_rot_pair(model, m, a)[1] for m in machs]
                                 for a in alphas], cg_x_m))

    cm_alphas = _nodes_alphas(model, 'cm')
    pitch.append(_product('aero/coefficient/CMa',
                          'pitch moment, nose-up positive (body +y)',
                          ['metrics/cbarw-ft'],
                          _matrix_table('aero/alpha-rad',
                                        [math.radians(a) for a in cm_alphas], machs,
                                        [[_at(model, 'cm', m, a, 0.0) for m in machs]
                                         for a in cm_alphas]), None))
    if cg_x_m is not None and 'beta_deg' in model.tables['cy'].axes:
        cy_betas = _nodes_betas(model, 'cy')
        yaw.append(_cg_cancel('CNcg', 'CG coupling of the side force, cancelled '
                              '(the document couples about the live CG)',
                              'aero/beta-rad',
                              [math.radians(b) for b in cy_betas], machs,
                              [[_at(model, 'cy', m, 0.0, b) for m in machs]
                               for b in cy_betas], cg_x_m))
    if 'beta_deg' in model.tables['cn_yaw'].axes:
        betas = _nodes_betas(model, 'cn_yaw')
        yaw.append(_product('aero/coefficient/CNb', 'yaw moment from sideslip (body +z)',
                            ['metrics/cbarw-ft'],
                            _matrix_table('aero/beta-rad',
                                          [math.radians(b) for b in betas], machs,
                                          [[_at(model, 'cn_yaw', m, 0.0, b) for m in machs]
                                           for b in betas]), None))

    if model.roll_damping_cl_p:
        roll.append(_product('aero/coefficient/Clp', 'roll damping',
                             ['metrics/bw-ft', 'aero/bi2vel', 'velocities/p-rad_sec'],
                             None, model.roll_damping_cl_p))
    if model.pitch_damping_cm_q:
        pitch.append(_product('aero/coefficient/CMq', 'pitch damping',
                              ['metrics/cbarw-ft', 'aero/ci2vel', 'velocities/q-rad_sec'],
                              None, model.pitch_damping_cm_q))
    if model.yaw_damping_cn_yaw_r:
        yaw.append(_product('aero/coefficient/CNr', 'yaw damping',
                            ['metrics/cbarw-ft', 'aero/ci2vel', 'velocities/r-rad_sec'],
                            None, model.yaw_damping_cn_yaw_r))

    if not drag:
        drag.append(_zero('CDzero', 'no axial force in this aero document'))
    if not lift:
        lift.append(_zero('CLzero', 'no normal force in this aero document'))
    if not side:
        side.append(_zero('CYzero', 'no side force in this aero document'))
    if not pitch:
        pitch.append(_zero('CMzero', 'no pitch moment in this aero document'))
    if not yaw:
        yaw.append(_zero('CNzero', 'no yaw moment in this aero document'))
    if not roll:
        roll.append(_zero('Clzero', 'no roll moment in this aero document'))

    out = ['<aerodynamics>',
           '  <!-- generated by orbitopt.flightlab.sixdof.jsbsim_aero from the '
           'Flight Lab aero document ' + model.name + '; regenerate instead of editing -->',
    ]
    out += _indented(_axis('DRAG', drag) + _axis('LIFT', lift) + _axis('SIDE', side)
                     + _axis('ROLL', roll) + _axis('PITCH', pitch) + _axis('YAW', yaw))
    out.append(_end('aerodynamics'))
    return '\n'.join(out)



def _cg_cancel(name: str, desc: str, row_var: str, row_nodes: Sequence[float],
               machs: Sequence[float], matrix: Sequence[Sequence[float]],
               aerorp_m: float) -> list[str]:
    """Cancel the couple JSBSim's fixed AERORP adds on top of the document's.

    The aero document applies its forces at the *live* CG and folds the
    static margin into ``cm``/``cn_yaw``; JSBSim applies its forces at a
    fixed reference point instead.  With ``AERORP`` at the dry CG and the CG
    travelling aft as propellant burns, the applied force carries an extra
    ``arm x force`` couple the document does not have, so the restoring
    stiffness of an axisymmetric body would depend on the propellant load --
    not the modelled physics.  This term subtracts exactly that couple,
    ``(cg - aerorp) / chord`` times the very force table it belongs to (the
    same bilinear surface JSBSim applies the force from), as a function of
    ``inertia/cg-x-in``.  The net moment is then the document's own
    CG-referenced one at every propellant load."""
    aerorp_in = aerorp_m * IN_PER_M
    out = [f'<function name="aero/coefficient/{name}">',
           f'  <description>{desc}</description>',
           '  <quotient>',
           '    <product>',
           '      <value>-1.0</value>']
    out += _indented(_matrix_table(row_var, row_nodes, machs, matrix), 2)
    out += ['      <difference>',
            '        <property>inertia/cg-x-in</property>',
            f'        <value>{_num(aerorp_in)}</value>',
            '      ' + _end('difference'),
            f'      <value>{_num(M_TO_FT)}</value>',
            '    ' + _end('product'),
            '    <property>metrics/cbarw-ft</property>',
            '  ' + _end('quotient'),
            _end('function')]
    return out


def _sub1(pattern: str, repl, text: str, what: str, flags: int = 0) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise AeroError(f'the {BASE_AIRCRAFT} aircraft XML does not look like the file this '
                        f'generator expects ({what} not found once)')
    return new


def _dry_cg_x_m(base_xml: str) -> float:
    m = re.search(r'<location name="CG"[^>]*>\s*<x>\s*([-+0-9.eE]+)', base_xml)
    if m is None:
        raise AeroError('the base aircraft XML has no CG location to put AERORP at')
    return float(m.group(1))


def production_aircraft_xml(model: AeroModel, *, name: str = PRODUCTION_AIRCRAFT,
                            base_xml: str | None = None) -> ProductionAircraft:
    """The production aircraft XML: the E4 airframe with the aero document's
    tables in place of its zero block, at the document's reference geometry.

    Masses, tanks, engine and the TVC FCS are inherited verbatim -- that is the
    point of deriving instead of authoring: the two engines then differ only
    in what their physics does with the same vehicle."""
    if base_xml is None:
        base_xml = base_aircraft_xml()
    cg_x = _dry_cg_x_m(base_xml)
    aerodynamics = aerodynamics_block(model, cg_x_m=cg_x)
    xml = _sub1(r'<aircraft name="' + BASE_AIRCRAFT + '"', '<aircraft name="' + name + '"',
                base_xml, 'aircraft name')
    xml = _sub1(r'<description>[^<]*</description>\n  </fileheader>',
                '<description>Production 6-DOF stage: the E4 spike airframe with the Flight '
                'Lab aero document (aero_model=' + model.name + ') compiled into its '
                'aerodynamics section (data/flightlab/aero/' + PRODUCTION_AERO_MODEL
                + '.yaml)</description>\n  </fileheader>',
                xml, 'fileheader description')
    xml = _sub1(r'  \* Aerodynamics:[^\n]*\n',
                '  * Aerodynamics: generated from the packaged aero document '
                + PRODUCTION_AERO_MODEL + '.yaml by\n'
                '    orbitopt.flightlab.sixdof.jsbsim_aero (E1-04, task #253). Regenerate\n'
                '    with python -m orbitopt.flightlab.sixdof.jsbsim_aero; the YAML is the\n'
                '    source, this file is a build artefact. AERORP sits at the dry CG: the\n'
                '    aero document couples about the (moving) CG, JSBSim applies forces at a\n'
                '    fixed reference point, and the difference is a disclosed, quantified\n'
                '    fidelity delta (docs/adr/flightlab-0007).\n',
                xml, 'aerodynamics comment')
    xml = _sub1(r'<wingarea  unit="M2">[^<]*</wingarea>',
                '<wingarea  unit="M2"> ' + _num(model.reference_area_m2) + ' </wingarea>',
                xml, 'wingarea')
    xml = _sub1(r'<wingspan  unit="M">[^<]*</wingspan>',
                '<wingspan  unit="M">   ' + _num(model.reference_span_m) + ' </wingspan>',
                xml, 'wingspan')
    xml = _sub1(r'<chord     unit="M">[^<]*</chord>',
                '<chord     unit="M">   ' + _num(model.reference_length_m) + ' </chord>',
                xml, 'chord (the pitch/yaw reference length)')
    xml = _sub1(r'(<location name="AERORP" unit="M"> <x>)[^<]*(</x>)',
                r'\g<1> ' + _num(cg_x) + r' \g<2>', xml, 'AERORP')
    xml = _sub1(r'<aerodynamics>.*</aerodynamics>',
                lambda _m: aerodynamics, xml, 'the aerodynamics section', re.S)
    return ProductionAircraft(name=name, xml=xml, aero_model=model.name, cg_x_m=cg_x)


def _jsbsim_root() -> Path:
    # Same lookup as jsbsim_driver.data_root(), without importing that module:
    # this one has to stay importable where the JSBSim bindings are not.
    return Path(str(resources.files('orbitopt') / 'data' / 'flightlab' / 'jsbsim'))


def base_aircraft_xml(name: str = BASE_AIRCRAFT) -> str:
    return (_jsbsim_root() / 'aircraft' / name / (name + '.xml')).read_text(encoding='utf-8')


def production_model(name: str = PRODUCTION_AERO_MODEL) -> AeroModel:
    """The aero document the production aircraft is compiled from."""
    return load_packaged_aero(name)


def packaged_production_aircraft(name: str = PRODUCTION_AIRCRAFT) -> str:
    """The generated XML as checked into the package."""
    return (_jsbsim_root() / 'aircraft' / name / (name + '.xml')).read_text(encoding='utf-8')


def aircraft_aero_note(name: str) -> str:
    """What a packaged aircraft's ``<aerodynamics>`` section really flies:
    ``"zero-aero"``, or the aero document its tables were compiled from.

    Same question :func:`orbitopt.flightlab.parity._aerodynamics_declared`
    answers for the parity gate, phrased for a run's assumptions list; the
    aircraft may declare an ``<aerodynamics>`` block that is all zeros, and
    that is still a ballistic model."""
    path = _jsbsim_root() / 'aircraft' / name / (name + '.xml')
    root = ET.parse(path).getroot()
    section = root.find('aerodynamics')
    if section is None:
        return 'zero-aero'
    if not section.findall('.//table'):
        if not any(node.text and abs(float(node.text)) > 0.0
                   for node in section.iter('value')):
            return 'zero-aero'
        return 'constant aerodynamic coefficients (hand-written, not generated)'
    match = re.search(r'aero_model=([\w.-]+)',
                      root.findtext('./fileheader/description', ''))
    source = f" compiled from the aero document '{match.group(1)}'" if match else ''
    return f'aerodynamic tables{source}'


def regenerate(*, check: bool = False, name: str = PRODUCTION_AIRCRAFT,
               model: str = PRODUCTION_AERO_MODEL) -> tuple[Path, bool]:
    """Rewrite (or verify) the packaged production aircraft from its document.

    ``check=True`` writes nothing and reports whether the checked-in file is
    what a fresh generation produces -- the test that keeps the artefact and
    the YAML from drifting apart."""
    aircraft = production_aircraft_xml(production_model(model), name=name)
    path = _jsbsim_root() / 'aircraft' / name / (name + '.xml')
    if check:
        return path, path.is_file() and path.read_text(encoding='utf-8') == aircraft.xml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(aircraft.xml, encoding='utf-8')
    return path, True


def _bilinear(rows: Sequence[float], cols: Sequence[float],
              matrix: Sequence[Sequence[float]], r: float, c: float) -> float:
    # Linear interpolation with clamping outside, matching JSBSim's tableData
    # behaviour on the grid the emitter wrote.
    def bracket(nodes, v):
        if v <= nodes[0]:
            return 0, 0.0
        if v >= nodes[-1]:
            return len(nodes) - 2, 1.0
        i = max(j for j in range(len(nodes) - 1) if nodes[j] <= v)
        return i, (v - nodes[i]) / (nodes[i + 1] - nodes[i])
    i, fr = bracket(list(rows), r)
    j, fc = bracket(list(cols), c)
    a = matrix[i][j] + (matrix[i][j + 1] - matrix[i][j]) * fc
    b = matrix[i + 1][j] + (matrix[i + 1][j + 1] - matrix[i + 1][j]) * fc
    return a + (b - a) * fr


def _grid(values: Sequence[float], n: int) -> list[float]:
    if len(values) < 2:
        return [float(values[0])]
    lo, hi = float(values[0]), float(values[-1])
    return [lo + (hi - lo) * k / (n - 1) for k in range(n)]


def promotion_fidelity(model: AeroModel, *, n_alpha: int = 9, n_beta: int = 5,
                       n_mach: int = 7) -> PromotionFidelity:
    """Where the generated JSBSim terms and the native model disagree.

    Both are read at the *same* interior states (nothing on a node, so the
    comparison is of the interpolation, not of the tabulated numbers). The
    stability-axis values JSBSim will use are rotated back into body axes with
    the inverse of the emitter's rotation, then differenced against the
    document."""
    ca_table, cy_table, cm_table = (model.tables['ca'], model.tables['cy'],
                                    model.tables['cm'])
    machs = _grid(_nodes(model, 'ca', 'mach'), n_mach)
    alphas = _grid(_nodes(model, 'ca', 'alpha_deg'), n_alpha)
    betas = (_grid(_nodes(model, 'ca', 'beta_deg'), n_beta)
             if 'beta_deg' in ca_table.axes else [0.0])

    def matrix(key: str, row_axis: str, cell):
        # (row nodes, column nodes, values) of the table the emitter would write.
        table = model.tables[key]
        nodes = list(_nodes(model, key, row_axis)) if row_axis in table.axes else [0.0]
        cols = list(_nodes(model, key, 'mach'))
        return nodes, cols, [[cell(m, r) for m in cols] for r in nodes]

    cd_rows, cd_cols, cd_val = matrix('ca', 'alpha_deg', lambda m, a: _rot_pair(model, m, a)[0])
    cl_rows, cl_cols, cl_val = matrix('ca', 'alpha_deg', lambda m, a: _rot_pair(model, m, a)[1])
    cdb_rows, cdb_cols, cdb_val = matrix('ca', 'beta_deg', lambda m, b: _cd_beta(model, m, b))
    cy_rows, cy_cols, cy_val = matrix('cy', 'beta_deg', lambda m, b: _at(model, 'cy', m, 0.0, b))
    cm_rows, cm_cols, cm_val = matrix('cm', 'alpha_deg', lambda m, a: _at(model, 'cm', m, a, 0.0))
    cnb_rows, cnb_cols, cnb_val = matrix('cn_yaw', 'beta_deg',
                                         lambda m, b: _at(model, 'cn_yaw', m, 0.0, b))

    worst = dict.fromkeys(('ca', 'cn', 'cy', 'cm', 'cn_yaw'), 0.0)
    samples = 0
    for mach in machs:
        for alpha in alphas:
            for beta in betas:
                drag = (_bilinear(cd_rows, cd_cols, cd_val, alpha, mach)
                        + _bilinear(cdb_rows, cdb_cols, cdb_val, beta, mach))
                lift = _bilinear(cl_rows, cl_cols, cl_val, alpha, mach)
                ca_rec = drag * math.cos(math.radians(alpha)) - lift * math.sin(math.radians(alpha))
                cn_rec = lift * math.cos(math.radians(alpha)) + drag * math.sin(math.radians(alpha))
                got = {'ca': ca_rec, 'cn': cn_rec,
                       'cy': _bilinear(cy_rows, cy_cols, cy_val, beta, mach),
                       'cm': _bilinear(cm_rows, cm_cols, cm_val, alpha, mach),
                       'cn_yaw': _bilinear(cnb_rows, cnb_cols, cnb_val, beta, mach)}
                for key, value in got.items():
                    worst[key] = max(worst[key], abs(value - _at(model, key, mach, alpha, beta)))
                samples += 1
    return PromotionFidelity(ca=worst['ca'], cn=worst['cn'], cy=worst['cy'],
                             cm=worst['cm'], cn_yaw=worst['cn_yaw'], samples=samples)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m orbitopt.flightlab.sixdof.jsbsim_aero [--check]``."""
    import argparse
    parser = argparse.ArgumentParser(prog='orbitopt.flightlab.sixdof.jsbsim_aero')
    parser.add_argument('--check', action='store_true',
                        help='report whether the packaged XML is current instead of writing it')
    args = parser.parse_args(argv)
    path, fresh = regenerate(check=args.check)
    if args.check:
        print(('up to date: ' if fresh else 'STALE (regenerate): ') + str(path))
        return 0 if fresh else 1
    print('wrote ' + str(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
