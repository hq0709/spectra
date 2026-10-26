"""Canonical event records (CER) compiled from MIMIC-IV ICU and hospital tables.

Rules (FINAL_PROPOSAL.md, Core Mechanism sections 4 and 9):
- Track A allowlist / denylist from resources/mimic_itemids.yaml; denied items never become inputs (Track A
  or Track B).
- Occurrence-time cutoff (primary): the state at t0 is carried forward; transitions and actions with a
  start in (t0, tk] enter; end times, durations and doses are clipped at tk; a continuous dose that cannot
  be reconstructed from documented rates is unknown. The availability cutoff (sensitivity analysis)
  additionally drops rows with storetime > tk.
- Interval rows compile to state + transitions: a line active from before t0 to t_end in (t0, tk] becomes
  "present (from before t0)" + "remove at t_end", a candidate removal (needs report corroboration) only
  when the line does not continue into the next department and t_end is strictly before ICU discharge;
  invasive ventilation ending inside the window becomes "ventilation ongoing" + "ventilation stop".
- Infusions are reconstructed as contiguous segments of one order (linkorderid); a documented pause ends
  a segment, so activity at t0 and in the window comes from the segments that overlap it.
- Track B adds charted vitals / respiratory / hemodynamic observations, outputs, laboratory values and
  microbiology (collection, then each result at its store time); observations that presuppose a device are
  tagged as that device's dependents.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .config import mimic_ontology_config
from .types import CER, Device

ONE_MINUTE = pd.Timedelta(minutes=1)
MICRO_COLUMNS = ["subject_id", "hadm_id", "micro_specimen_id", "chartdate", "charttime", "spec_type_desc", "test_name",
                 "storedate", "storetime", "org_name", "comments"]
MASS = {"mcg": 1e-3, "mg": 1.0, "grams": 1e3, "gm": 1e3, "g": 1e3}


def _hours(t: pd.Timestamp, t0: pd.Timestamp) -> float:
    return (t - t0).total_seconds() / 3600.0


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (x is pd.NaT) or (not isinstance(x, str) and pd.isna(x))


@dataclass
class MimicOntology:
    cfg: dict = field(default_factory=mimic_ontology_config)

    def __post_init__(self):
        pe = self.cfg["procedureevents"]
        self.device_point = {int(k): v for k, v in pe["device_point"].items()}
        self.device_interval = {int(k): v for k, v in pe["device_interval"].items()}
        self.interval_action = {int(k): v for k, v in pe["interval_action"].items()}
        self.exclusion = {int(k): v for k, v in pe["exclusion"].items()}
        self.pe_allow_categories = set(pe["allow_categories"])
        ie = self.cfg["inputevents"]
        self.ie_allow_categories = set(ie["allow_categories"])
        self.ie_type_by_category = dict(ie["action_type_by_category"])
        self.priority_by_action_type = dict(ie["priority_by_action_type"])
        self.medication_class = {int(i): cls for cls, ids in ie["medication_classes"].items() for i in ids}
        self.vent_settings = {int(k): v for k, v in self.cfg["chartevents"]["ventilator_settings"].items()}
        deny = self.cfg["deny"]
        self.deny_itemids = {int(i) for i in deny["itemids"]}
        self.deny_pe_categories = set(deny["procedureevents_categories"])
        self.deny_chart_patterns = [re.compile(p) for p in deny["chartevents_label_patterns"]]
        self.location_map = dict(self.cfg["location_map"])
        tb = self.cfg["track_b"]
        self.track_b_chart_categories = set(tb["chartevents_categories"])
        self.device_dependents = tb["device_dependents"]
        self.priority_order = list(self.cfg["priority_order"])

    def denied(self, itemid: int, label: str = "", category: str = "", table: str = "") -> bool:
        if int(itemid) in self.deny_itemids:
            return True
        if table == "procedureevents" and category in self.deny_pe_categories:
            return True
        if table == "chartevents" and any(p.search(str(label)) for p in self.deny_chart_patterns):
            return True
        return False

    def site_side(self, location) -> tuple:
        if _nan(location) or str(location).strip() == "":
            return "none", "none"
        m = self.location_map.get(str(location).strip())
        if m is None:
            return "other", "none"
        return m["site"], m["side"]

    def input_action_type(self, itemid: int, category: str) -> str:
        cls = self.medication_class.get(int(itemid))
        return cls if cls is not None else self.ie_type_by_category.get(category, "other_medication")

    def priority_of(self, action_type: str) -> str:
        if action_type.startswith("ventilator_setting"):
            return "ventilator_setting"
        return self.priority_by_action_type.get(action_type, "other_medication")

    def dependent_of(self, table: str, itemid: int, category: str) -> Optional[str]:
        """The device whose presence a Track B observation presupposes (withheld from that device's tests)."""
        iid = int(itemid)
        for dev, spec in self.device_dependents.items():
            if table == "chartevents":
                if iid in set(spec.get("chartevents", []) or []):
                    return dev
                if category in set(spec.get("chartevents_categories", []) or []) and \
                        iid not in set(spec.get("chartevents_general", []) or []):
                    return dev
            if table == "outputevents" and iid in set(spec.get("outputevents", []) or []):
                return dev
        return None


def _read_csv(path: Path, parse: Sequence[str], usecols=None, filt=None, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Chunked read keeping only filtered rows; the filter is applied per chunk, so memory is bounded by the
    retained rows (the cohort's stays), not by the table."""
    if not path.exists():
        alt = path.with_suffix("") if path.suffix == ".gz" else Path(str(path) + ".gz")
        path = alt
    # only an empty field is missing: MIMIC charts real values such as "None" (ectopy type) and "NA"
    na = {"keep_default_na": False, "na_values": [""]}
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False, **na):
        if filt is not None:
            chunk = filt(chunk)
        if len(chunk):
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.read_csv(path, usecols=usecols, nrows=0, **na)
    for c in parse:
        if c in df:
            df[c] = pd.to_datetime(df[c], format="ISO8601")  # some MIMIC timestamps carry fractional seconds
    return df


@dataclass
class IcuTables:
    """MIMIC-IV tables restricted to a cohort of ICU stays. Build once per cohort; `for_stay` slices are
    indexed once, so compiling many windows never re-filters the whole tables."""

    procedureevents: pd.DataFrame
    inputevents: pd.DataFrame
    chartevents: pd.DataFrame
    d_items: pd.DataFrame
    icustays: pd.DataFrame
    outputevents: Optional[pd.DataFrame] = None
    labevents: Optional[pd.DataFrame] = None
    microbiology: Optional[pd.DataFrame] = None
    d_labitems: Optional[pd.DataFrame] = None
    track_b: bool = False
    _index: Dict[int, "IcuTables"] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dir(cls, icu_dir: str | Path, hosp_dir: Optional[str | Path] = None, stay_ids: Optional[Iterable] = None,
                 ontology: Optional[MimicOntology] = None, track_b: bool = False) -> "IcuTables":
        icu_dir = Path(icu_dir)
        hosp_dir = Path(hosp_dir) if hosp_dir else icu_dir.parent / "hosp"
        ont = ontology or MimicOntology()
        stays = None if stay_ids is None else {int(s) for s in stay_ids}
        by_stay = (lambda d: d[d["stay_id"].isin(stays)]) if stays is not None else None
        times = ["starttime", "endtime", "storetime"]
        d_items = _read_csv(icu_dir / "d_items.csv.gz", [])
        icustays = _read_csv(icu_dir / "icustays.csv.gz", ["intime", "outtime"], filt=by_stay)
        chart_items = set(ont.vent_settings)
        if track_b:
            cats = ont.track_b_chart_categories
            chart_items |= set(d_items.loc[(d_items["linksto"] == "chartevents") & d_items["category"].isin(cats), "itemid"].astype(int))
        chart_filter = (lambda d: d[d["itemid"].isin(chart_items) & (d["stay_id"].isin(stays) if stays is not None else True)])
        pe = _read_csv(icu_dir / "procedureevents.csv.gz", times, filt=by_stay)
        ie = _read_csv(icu_dir / "inputevents.csv.gz", times, filt=by_stay)
        ce = _read_csv(icu_dir / "chartevents.csv.gz", ["charttime", "storetime"], filt=chart_filter)
        oe = lab = micro = dlab = None
        if track_b:
            oe = _read_csv(icu_dir / "outputevents.csv.gz", ["charttime", "storetime"], filt=by_stay)
            subjects = set(icustays["subject_id"].astype(int))
            by_subj = lambda d: d[d["subject_id"].isin(subjects)]
            lab = _read_csv(hosp_dir / "labevents.csv.gz", ["charttime", "storetime"], filt=by_subj,
                            usecols=["subject_id", "hadm_id", "itemid", "charttime", "storetime", "value", "valuenum", "valueuom"])
            micro = _read_csv(hosp_dir / "microbiologyevents.csv.gz", ["chartdate", "charttime", "storedate", "storetime"], filt=by_subj,
                              usecols=MICRO_COLUMNS)
            dlab = _read_csv(hosp_dir / "d_labitems.csv.gz", [])
        return cls(pe, ie, ce, d_items, icustays, oe, lab, micro, dlab, track_b)

    @classmethod
    def from_duckdb(cls, icu_dir: str | Path, hosp_dir: Optional[str | Path] = None, stay_ids: Iterable = (),
                    ontology: Optional[MimicOntology] = None, track_b: bool = False) -> "IcuTables":
        """Bounded reads for full MIMIC-IV on the MIMIC server: DuckDB scans the compressed CSVs and returns
        only the cohort's stays (and, for chartevents, only the needed items)."""
        import duckdb

        icu_dir = Path(icu_dir)
        hosp_dir = Path(hosp_dir) if hosp_dir else icu_dir.parent / "hosp"
        ont = ontology or MimicOntology()
        stays = sorted({int(s) for s in stay_ids})
        if not stays:
            raise ValueError("from_duckdb needs the cohort's stay ids (bounded reads)")
        con = duckdb.connect()
        con.execute("CREATE TEMP TABLE cohort(stay_id BIGINT)")
        con.executemany("INSERT INTO cohort VALUES (?)", [(s,) for s in stays])

        def path(d: Path, name: str) -> str:
            p = d / f"{name}.csv.gz"
            return str(p if p.exists() else d / f"{name}.csv")

        def q(sql: str, parse: Sequence[str]) -> pd.DataFrame:
            df = con.execute(sql).df()
            for c in parse:
                if c in df and not pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = pd.to_datetime(df[c], format="ISO8601")
            return df

        times = ["starttime", "endtime", "storetime"]
        d_items = q(f"SELECT * FROM read_csv_auto('{path(icu_dir, 'd_items')}')", [])
        icustays = q(f"SELECT * FROM read_csv_auto('{path(icu_dir, 'icustays')}') WHERE stay_id IN (SELECT stay_id FROM cohort)",
                     ["intime", "outtime"])
        chart_items = set(ont.vent_settings)
        if track_b:
            cats = ont.track_b_chart_categories
            chart_items |= set(d_items.loc[(d_items["linksto"] == "chartevents") & d_items["category"].isin(cats), "itemid"].astype(int))
        items = ",".join(str(i) for i in sorted(chart_items))
        in_cohort = "stay_id IN (SELECT stay_id FROM cohort)"
        pe = q(f"SELECT * FROM read_csv_auto('{path(icu_dir, 'procedureevents')}') WHERE {in_cohort}", times)
        ie = q(f"SELECT * FROM read_csv_auto('{path(icu_dir, 'inputevents')}') WHERE {in_cohort}", times)
        ce = q(f"SELECT * FROM read_csv_auto('{path(icu_dir, 'chartevents')}') WHERE {in_cohort} AND itemid IN ({items})",
               ["charttime", "storetime"])
        oe = lab = micro = dlab = None
        if track_b:
            oe = q(f"SELECT * FROM read_csv_auto('{path(icu_dir, 'outputevents')}') WHERE {in_cohort}", ["charttime", "storetime"])
            subj = ",".join(str(int(s)) for s in sorted(set(icustays["subject_id"])))
            lab = q(f"SELECT subject_id, hadm_id, itemid, charttime, storetime, value, valuenum, valueuom FROM "
                    f"read_csv_auto('{path(hosp_dir, 'labevents')}') WHERE subject_id IN ({subj})", ["charttime", "storetime"])
            micro = q(f"SELECT {', '.join(MICRO_COLUMNS)} FROM "
                      f"read_csv_auto('{path(hosp_dir, 'microbiologyevents')}') WHERE subject_id IN ({subj})",
                      ["chartdate", "charttime", "storedate", "storetime"])
            dlab = q(f"SELECT * FROM read_csv_auto('{path(hosp_dir, 'd_labitems')}')", [])
        con.close()
        return cls(pe, ie, ce, d_items, icustays, oe, lab, micro, dlab, track_b)

    def for_stay(self, stay_id) -> "IcuTables":
        s = int(stay_id)
        if not self._index:
            groups = {}
            for name in ("procedureevents", "inputevents", "chartevents", "outputevents", "icustays"):
                df = getattr(self, name)
                groups[name] = {} if df is None else {int(k): g for k, g in df.groupby("stay_id")}
            stays = set().union(*[set(v) for v in groups.values()])
            subj = dict(zip(self.icustays["stay_id"].astype(int), self.icustays["subject_id"].astype(int)))
            lab_g = {} if self.labevents is None else {int(k): g for k, g in self.labevents.groupby("subject_id")}
            mic_g = {} if self.microbiology is None else {int(k): g for k, g in self.microbiology.groupby("subject_id")}
            empty = lambda df: None if df is None else df.iloc[0:0]
            for st in stays:
                g = lambda n: groups[n].get(st, empty(getattr(self, n)))
                self._index[st] = IcuTables(g("procedureevents"), g("inputevents"), g("chartevents"), self.d_items,
                                            g("icustays"), g("outputevents") if self.outputevents is not None else None,
                                            lab_g.get(subj.get(st), empty(self.labevents)) if self.labevents is not None else None,
                                            mic_g.get(subj.get(st), empty(self.microbiology)) if self.microbiology is not None else None,
                                            self.d_labitems, self.track_b)
        if s not in self._index:
            raise KeyError(f"stay {stay_id} is not in this cohort")
        return self._index[s]

    def outtime(self, stay_id) -> Optional[pd.Timestamp]:
        st = self.for_stay(stay_id).icustays
        return None if st is None or st.empty else st["outtime"].iloc[0]


class CERCompiler:
    """Compile one query window (t0, tk] of one ICU stay into canonical event records."""

    def __init__(self, ontology: Optional[MimicOntology] = None):
        self.ont = ontology or MimicOntology()

    def compile(self, tables: IcuTables, stay_id, t0, tk, track: str = "A", cutoff: str = "occurrence") -> List[CER]:
        if track not in ("A", "B"):
            raise ValueError("track must be 'A' or 'B' (Track Delta carries no compiled stream)")
        if cutoff not in ("occurrence", "availability"):
            raise ValueError("cutoff must be 'occurrence' or 'availability'")
        if track == "B" and not tables.track_b:
            raise ValueError("these tables were loaded without Track B sources")
        t0, tk = pd.Timestamp(t0), pd.Timestamp(tk)
        if not tk > t0:
            raise ValueError("window end must follow the prior film")
        st = tables.for_stay(stay_id)
        labels = dict(zip(tables.d_items["itemid"].astype(int), tables.d_items["label"]))
        cats = dict(zip(tables.d_items["itemid"].astype(int), tables.d_items["category"]))
        discharge = tables.outtime(stay_id)
        out: List[CER] = []
        out += self._procedures(st.procedureevents, t0, tk, cutoff, labels, cats, discharge)
        out += self._inputs(st.inputevents, t0, tk, cutoff, labels, cats)
        out += self._vent_settings(st.chartevents, t0, tk, cutoff, labels)
        if track == "B":
            out += self._observations(st, t0, tk, cutoff, labels, cats)
        out.sort(key=lambda c: c.sort_key())
        return out

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _available(row, tk, cutoff) -> bool:
        if cutoff != "availability":
            return True
        stt = row.get("storetime")
        return _nan(stt) or stt <= tk

    def _procedures(self, df, t0, tk, cutoff, labels, cats, discharge) -> List[CER]:
        out: List[CER] = []
        if df is None or df.empty:
            return out
        for row in df.sort_values("starttime").to_dict("records"):
            iid = int(row["itemid"])
            cat = cats.get(iid, "")
            if self.ont.denied(iid, labels.get(iid, ""), cat, "procedureevents"):
                continue
            if not self._available(row, tk, cutoff):
                continue
            start, end = row["starttime"], row.get("endtime")
            cont = row.get("continueinnextdept")
            continues = (not _nan(cont)) and int(cont) == 1
            if iid in self.ont.device_point:
                spec = self.ont.device_point[iid]
                if t0 < start <= tk:
                    dev = Device(spec["device"], spec["transition"], spec.get("site", "none"), "none", labels.get(iid, ""))
                    out.append(CER("transition", spec["action_type"], _hours(start, t0), device=dev,
                                   source_itemids=(iid,), priority="device_procedure"))
            elif iid in self.ont.device_interval:
                out += self._device_interval(row, iid, t0, tk, continues, discharge, labels)
            elif iid in self.ont.interval_action:
                out += self._interval_action(self.ont.interval_action[iid]["action_type"], iid, start, end, t0, tk)
            elif iid in self.ont.exclusion:
                spec = self.ont.exclusion[iid]
                if start <= t0:
                    out.append(CER("state", spec["action_type"], None, source_itemids=(iid,), priority="device_procedure"))
                elif start <= tk:
                    out.append(CER("action", spec["action_type"], _hours(start, t0), source_itemids=(iid,),
                                   priority="device_procedure"))
            elif cat in self.ont.pe_allow_categories:
                atype = _slug(labels.get(iid, iid))
                has_end = not _nan(end) and end > start + ONE_MINUTE
                if start <= t0 and has_end and end > t0:
                    out.append(CER("state", atype, None, None if end > tk else _hours(end, t0), end > tk,
                                   source_itemids=(iid,), priority="device_procedure"))
                elif t0 < start <= tk:
                    t_end = _hours(min(end, tk), t0) if has_end else None
                    out.append(CER("action", atype, _hours(start, t0), t_end, has_end and end > tk, source_itemids=(iid,),
                                   priority="device_procedure"))
        return out

    def _device_interval(self, row, iid, t0, tk, continues, discharge, labels) -> List[CER]:
        spec = self.ont.device_interval[iid]
        site, side = self.ont.site_side(row.get("location"))
        dev_type, subtype = spec["device"], spec.get("subtype", labels.get(iid, ""))
        start, end = row["starttime"], row.get("endtime")
        has_end = not _nan(end)
        # an endpoint is a candidate removal only if the line did not continue elsewhere and the endpoint is
        # strictly before ICU discharge (otherwise it marks the end of documentation, not a removal)
        removal_candidate = has_end and not continues and (discharge is None or end < discharge)
        out: List[CER] = []
        state_dev = Device(dev_type, "none", site, side, subtype)
        if start <= t0 and (not has_end or end > t0):
            removed_in_window = removal_candidate and end <= tk
            out.append(CER("state", "line_present" if dev_type in ("CVC", "PICC") else f"{dev_type}_present", None,
                           ongoing_at_delta=not removed_in_window, device=state_dev, source_itemids=(iid,),
                           priority="device_procedure"))
            if removed_in_window:
                out.append(CER("transition", "line_removal", _hours(end, t0),
                               device=Device(dev_type, "remove", site, side, subtype), source_itemids=(iid,),
                               priority="device_procedure", candidate=True))
        elif t0 < start <= tk:
            out.append(CER("transition", "line_placement", _hours(start, t0),
                           device=Device(dev_type, "place", site, side, subtype), source_itemids=(iid,),
                           priority="device_procedure"))
            if removal_candidate and end <= tk:
                out.append(CER("transition", "line_removal", _hours(end, t0),
                               device=Device(dev_type, "remove", site, side, subtype), source_itemids=(iid,),
                               priority="device_procedure", candidate=True))
        return out

    @staticmethod
    def _interval_action(atype, iid, start, end, t0, tk) -> List[CER]:
        has_end = not _nan(end)
        out: List[CER] = []
        if start <= t0 and (not has_end or end > t0):
            stops = has_end and end <= tk
            out.append(CER("state", atype, None, ongoing_at_delta=not stops, source_itemids=(iid,), priority="device_procedure"))
            if stops:
                out.append(CER("action", f"{atype}_stop", _hours(end, t0), source_itemids=(iid,), priority="device_procedure"))
        elif t0 < start <= tk:
            stops = has_end and end <= tk
            out.append(CER("action", atype, _hours(start, t0), ongoing_at_delta=not stops, source_itemids=(iid,),
                           priority="device_procedure"))
            if stops:
                out.append(CER("action", f"{atype}_stop", _hours(end, t0), source_itemids=(iid,), priority="device_procedure"))
        return out

    # -- infusions -------------------------------------------------------------------------
    @staticmethod
    def _segments(rows: List[dict]) -> List[List[dict]]:
        """Contiguous segments of one order: a gap longer than one minute (a documented pause) ends a segment."""
        rows = sorted(rows, key=lambda r: r["starttime"])
        segs, cur = [], [rows[0]]
        for r in rows[1:]:
            prev_end = cur[-1]["endtime"]
            if not _nan(prev_end) and r["starttime"] <= prev_end + ONE_MINUTE:
                cur.append(r)
            else:
                segs.append(cur)
                cur = [r]
        segs.append(cur)
        return segs

    @staticmethod
    def _rate_amount(r: dict, lo: pd.Timestamp, hi: pd.Timestamp) -> Optional[float]:
        """Amount delivered over [lo, hi] from the documented rate, in the row's amount unit; None if the rate
        or its units cannot reconstruct it."""
        rate, ru, au = r.get("rate"), r.get("rateuom"), r.get("amountuom")
        if _nan(rate) or _nan(ru) or _nan(au):
            return None
        parts = str(ru).replace(" ", "").split("/")
        if len(parts) not in (2, 3):
            return None
        num, per = parts[0].lower(), parts[-1].lower()
        per_h = {"hour": 1.0, "hr": 1.0, "min": 60.0}.get(per)
        if per_h is None:
            return None
        amt_u = str(au).lower().rstrip(".")
        num = num.rstrip(".")
        if num == amt_u:
            factor = 1.0
        elif num in MASS and amt_u in MASS:
            factor = MASS[num] / MASS[amt_u]
        else:
            return None
        per_kg = 1.0
        if len(parts) == 3:
            if parts[1].lower() != "kg" or _nan(r.get("patientweight")):
                return None
            per_kg = float(r["patientweight"])
        hours = (hi - lo).total_seconds() / 3600.0
        return float(rate) * per_h * per_kg * factor * hours

    def _dose(self, rows: List[dict], t0, tk) -> Optional[float]:
        """Amount inside (t0, tk]: rows wholly inside count whole; a row crossing a window boundary is
        reconstructed from its documented rate, otherwise the dose is unknown."""
        total = 0.0
        for r in rows:
            s, e, amt = r["starttime"], r.get("endtime"), r.get("amount")
            if _nan(e):
                return None
            lo, hi = max(s, t0), min(e, tk)
            point = s == e
            if point:
                if t0 < s <= tk:
                    if _nan(amt):
                        return None
                    total += float(amt)
                continue
            if hi <= lo:
                continue
            if s > t0 and e <= tk:
                if _nan(amt):
                    return None
                total += float(amt)
            else:
                part = self._rate_amount(r, lo, hi)
                if part is None:
                    return None
                total += part
        return round(total, 6)

    def _inputs(self, df, t0, tk, cutoff, labels, cats) -> List[CER]:
        out: List[CER] = []
        if df is None or df.empty:
            return out
        df = df[df["itemid"].map(lambda i: cats.get(int(i), "") in self.ont.ie_allow_categories)]
        if df.empty:
            return out
        key = "linkorderid" if "linkorderid" in df else "orderid"
        for (iid, _link), grp in df.groupby(["itemid", key], sort=False):
            iid = int(iid)
            if self.ont.denied(iid, labels.get(iid, ""), cats.get(iid, ""), "inputevents"):
                continue
            rows = [r for r in grp.to_dict("records") if self._available(r, tk, cutoff)]
            if not rows:
                continue
            atype = self.ont.input_action_type(iid, cats.get(iid, ""))
            prio = self.ont.priority_of(atype)
            unit = rows[0].get("amountuom")
            for seg in self._segments(rows):
                s0 = seg[0]["starttime"]
                ends = [r["endtime"] for r in seg]
                s1 = pd.NaT if any(_nan(e) for e in ends) else max(ends)
                ongoing = _nan(s1) or s1 > tk
                if s0 <= t0 and (_nan(s1) or s1 > t0):
                    dose = self._dose(seg, t0, tk)
                    t_end = None if ongoing else _hours(s1, t0)   # a carry-in segment that stops in the window
                    out.append(CER("state", atype, None, t_end, ongoing_at_delta=ongoing, dose_value=dose,
                                   dose_unit=unit if dose is not None else None, value=labels.get(iid, ""),
                                   source_itemids=(iid,), priority=prio))
                elif t0 < s0 <= tk:
                    dose = self._dose(seg, t0, tk)
                    t_end = None if _nan(s1) else _hours(min(s1, tk), t0)
                    out.append(CER("action", atype, _hours(s0, t0), t_end, ongoing_at_delta=ongoing, dose_value=dose,
                                   dose_unit=unit if dose is not None else None, value=labels.get(iid, ""),
                                   source_itemids=(iid,), priority=prio))
        return out

    def _vent_settings(self, df, t0, tk, cutoff, labels) -> List[CER]:
        out: List[CER] = []
        if df is None or df.empty:
            return out
        df = df[df["itemid"].isin(list(self.ont.vent_settings))]
        for iid, grp in df.sort_values("charttime").groupby("itemid", sort=False):
            iid = int(iid)
            if self.ont.denied(iid, labels.get(iid, ""), "", "chartevents"):
                continue
            rows = [r for r in grp.to_dict("records") if self._available(r, tk, cutoff)]
            label = self.ont.vent_settings[iid]
            before = [r for r in rows if r["charttime"] <= t0]
            last = None
            if before:
                last = str(before[-1]["value"]).strip()
                out.append(CER("state", "ventilator_setting", None, value=f"{label}={last}", source_itemids=(iid,),
                               priority="ventilator_setting"))
            for r in rows:
                if t0 < r["charttime"] <= tk:
                    v = str(r["value"]).strip()
                    if v != last:  # value changes only
                        out.append(CER("action", "ventilator_setting", _hours(r["charttime"], t0), value=f"{label}={v}",
                                       source_itemids=(iid,), priority="ventilator_setting"))
                        last = v
        return out

    # -- Track B -----------------------------------------------------------------------------
    def _observations(self, st: IcuTables, t0, tk, cutoff, labels, cats) -> List[CER]:
        out: List[CER] = []

        def add(table, iid, t, value, atype, avail_row):
            if not (t0 < t <= tk) or not self._available(avail_row, tk, cutoff):
                return
            owner = self.ont.dependent_of(table, iid, cats.get(iid, "")) if table in ("chartevents", "outputevents") else None
            out.append(CER("observation", atype, _hours(t, t0), value=None if _nan(value) else str(value),
                           source_itemids=(int(iid),), priority="other_medication", track="B",
                           origin="record" if owner is None else f"dependent:{owner}"))

        ce = st.chartevents
        if ce is not None and not ce.empty:
            obs = ce[~ce["itemid"].isin(list(self.ont.vent_settings))]
            for r in obs.to_dict("records"):
                iid = int(r["itemid"])
                if self.ont.denied(iid, labels.get(iid, ""), cats.get(iid, ""), "chartevents"):
                    continue
                add("chartevents", iid, r["charttime"], r.get("value"), f"obs:{_slug(labels.get(iid, iid))}", r)
        oe = st.outputevents
        if oe is not None and not oe.empty:
            for r in oe.to_dict("records"):
                iid = int(r["itemid"])
                if self.ont.denied(iid, labels.get(iid, ""), cats.get(iid, ""), "outputevents"):
                    continue
                add("outputevents", iid, r["charttime"], r.get("value"), f"obs:{_slug(labels.get(iid, iid))}", r)
        lab = st.labevents
        if lab is not None and not lab.empty:
            ll = {} if st.d_labitems is None else dict(zip(st.d_labitems["itemid"].astype(int), st.d_labitems["label"]))
            for r in lab.to_dict("records"):
                iid = int(r["itemid"])
                v = r.get("valuenum") if not _nan(r.get("valuenum")) else r.get("value")
                unit = "" if _nan(r.get("valueuom")) else f" {r['valueuom']}"
                add("labevents", iid, r["charttime"], None if _nan(v) else f"{v}{unit}", f"obs:lab:{_slug(ll.get(iid, iid))}", r)
        mic = st.microbiology
        if mic is not None and not mic.empty:
            # one row per organism and antibiotic: each specimen's collection and each distinct result enter once.
            # The collection enters at charttime (known when collected); the result is a later event at its
            # store time (a store date without time counts from the end of that day), under either cutoff.
            seen = set()
            for r in mic.to_dict("records"):
                spec = r.get("micro_specimen_id")
                atype = _slug(r.get("spec_type_desc") or "specimen")
                test = None if _nan(r.get("test_name")) else str(r["test_name"])
                t = r["charttime"] if not _nan(r.get("charttime")) else r.get("chartdate")
                if not _nan(t) and ("c", spec, test) not in seen:
                    seen.add(("c", spec, test))
                    add("microbiology", 0, t, test, f"obs:micro:{atype}", {})
                st_t, st_d = r.get("storetime"), r.get("storedate")
                t_res = st_t if not _nan(st_t) else (st_d + pd.Timedelta(days=1) if not _nan(st_d) else None)
                if t_res is None:
                    continue
                res = _micro_result(r)
                if ("r", spec, test, res) not in seen:
                    seen.add(("r", spec, test, res))
                    add("microbiology", 0, t_res, res, f"obs:micro_result:{atype}", {"storetime": t_res})
        return out


def _micro_result(r: dict) -> str:
    """The organism, else the laboratory's short result comment (e.g. "NO GROWTH."), else 'no organism reported'."""
    if not _nan(r.get("org_name")):
        return str(r["org_name"])
    c = r.get("comments")
    if _nan(c) or not str(c).strip() or str(c).strip().startswith("___"):
        return "no organism reported"
    first = re.split(r"(?<=\.)\s", " ".join(str(c).split()), maxsplit=1)[0]
    return first[:60]


def device_cers(stream: Sequence[CER], device: str) -> List[CER]:
    return [c for c in stream if c.device.type == device]
