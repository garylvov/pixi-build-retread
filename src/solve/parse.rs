use regex::Regex;

/// One `<package> <version-specifier>` clause found anywhere in a failed
/// solve's error text by the generic fallback extractor
/// ([`RegexConflictParser::extract_generic_mentions`]) -- the raw
/// material the ownership-driven fallback repair engine classifies and
/// generates candidates from, when none of the specific per-shape
/// `Conflict` parsers above matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mention {
    /// The mentioned package's name, as it appeared in the prose
    /// (case as-written; callers should canonicalize before comparing).
    pub package: String,
    /// The version specifier exactly as found, normalized to a
    /// conda/PEP440-compatible operator (`==`/`>=`/`<=`/`>`/`<`, `=`
    /// normalized to `==`), e.g. `"==1.38.0"`, `">=68,<81"`,
    /// `"==4.9.*"`.
    pub spec: String,
    /// The nearest preceding "<requirer> [==ver] would require" /
    /// "because <requirer>==ver depends on" clause's subject, when one
    /// was found before this mention in the text. Best-effort and
    /// informational only -- `None` when no such clause precedes it.
    pub requirer: Option<String>,
}

/// One actionable fact extracted from a failed solve.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum Conflict {
    /// A previously injected `==version` pin has no candidate in the ecosystem just tried.
    NoCandidates { package: String, version: String },
    /// conda-pinned version clashes with a pypi requirement.
    CondaBoundary { package: String, version: String },
    /// Conflict entirely inside the pypi subgraph.
    PypiInternal { package: String, version: String },
    /// A pypi-side requirement needs a range that a conda-sourced pin blocks.
    CondaWidenNeeded {
        package: String,
        op: String,
        floor: String,
        conda_version: String,
        /// Bounded "Because X==V depends on Y" walk up from `package`
        /// (the conda-provenance footer's, i.e. the transitive SYMPTOM
        /// package) toward the packages that actually require it,
        /// nearest first (e.g. `["torch", "isaacsim-core", ...]` for a
        /// `cuda-bindings` footer). Empty when the parse shape doesn't
        /// carry a uv "Because" chain (older conda-solver prose shapes).
        /// Lets repair re-attribute a conflict to a package with a real
        /// user conda pin when the footer package has none of its own.
        requiring_chain: Vec<String>,
        /// Fix #20: the pack/bundle name this conflict was raised inside
        /// (`computing uv closure for bundle `<name>`:`), when the
        /// conflict came from a backend uv-closure JSON-RPC error --
        /// `None` for the older direct conda-solver prose shapes (which
        /// aren't scoped to any one pack's closure). Doubles as the
        /// provenance signal: `Some` means "this is a backend-closure
        /// conflict", so repair must write any auto-override into the
        /// `.retread/overrides.json` ledger keyed by this bundle instead
        /// of a workspace manifest table the closure never reads.
        pack_name: Option<String>,
    },
    /// A uv conflict entirely inside the pypi subgraph where the losing
    /// side is an EXACT pin, with NO conda constraint attributable at all
    /// (the `no_conda_constraint_named` footer fired, unlike
    /// `CondaWidenNeeded`'s `conda_provenance` shape) -- e.g. a `deps-from`
    /// requirements file pins both `wandb==0.23.0` (whose metadata
    /// requires `sentry-sdk>=2.0.0`) and `sentry-sdk==1.38.0` directly.
    /// Doctrine: `retread-deps-from` pins are upstream advisories, not
    /// user intent, so `repair` may relax `package`'s exact pin to a
    /// `floor` -- but ONLY after confirming (at repair time, see
    /// `deps_from_owns_exact_pin` in repair.rs) that the pin actually
    /// originates from `pack_name`'s own deps-from source(s); this variant
    /// alone never implies that.
    DepsFromPin {
        /// The exact-pinned package blocking the other requirement (e.g.
        /// `sentry-sdk`).
        package: String,
        /// The pin's own exact version (e.g. `1.38.0`), informational
        /// (ledger old_spec / ownership double-check).
        pinned_version: String,
        /// Comparison operator of the unmet requirement (`>=`/`>`).
        op: String,
        /// Floor version demanded by the other package (e.g. `2.0.0`).
        floor: String,
        /// The package whose requirement collided with the pin (e.g.
        /// `wandb`); informational only.
        requirer: String,
        /// Bundle the uv closure was computed for. This shape only
        /// appears inside a backend uv-closure JSON-RPC error, so this is
        /// always `Some` in practice; `None` is treated as unrepairable.
        pack_name: Option<String>,
    },
    /// A `retread-deps-from` root's own EXACT pin (e.g. `hydra-core==1.3.2`)
    /// hard-depends on another package via a wildcard/exact range (e.g.
    /// `antlr4-python3-runtime==4.9.*`) for which NO version in that range
    /// has a usable wheel under `--no-build` (source-dist only) -- distinct
    /// from `DepsFromPin`: there is no COMPETING requirement from a sibling
    /// package, the pinned transitive dependency is simply unbuildable from
    /// a wheel. This shape only reaches `retread solve`'s parser at all
    /// when the backend's sdist-only self-heal ladder
    /// (`uv_closure::with_sdist_heal`: conda-route -> sdist auto-build)
    /// already tried and exhausted BOTH rungs for `package` -- i.e. the
    /// original uv prose survives verbatim with the ladder's own
    /// exhaustion guidance appended. Doctrine (matching `DepsFromPin`):
    /// `retread-deps-from` pins are upstream advisories, so `repair` may
    /// relax the REQUIRER's (`hydra-core`) own exact pin to a `>=` floor,
    /// letting uv pick a newer requirer release whose own metadata may no
    /// longer wildcard-pin the wheel-less package -- but ONLY after
    /// confirming (at repair time, see `deps_from_owns_exact_pin` in
    /// repair.rs) that the requirer's pin actually originates from
    /// `pack_name`'s own deps-from source(s).
    NoWheelTransitive {
        /// The wheel-less transitive package (e.g.
        /// `antlr4-python3-runtime`), informational.
        package: String,
        /// The unsatisfiable version range uv named for `package` (e.g.
        /// `>=4.9,<=4.9.3`), informational.
        range: String,
        /// The package whose exact pin wildcard-pins `package` (e.g.
        /// `hydra-core`) -- this is what repair relaxes.
        requirer: String,
        /// `requirer`'s own exact pinned version (e.g. `1.3.2`).
        requirer_pin: String,
        /// Bundle the uv closure was computed for; `None` is treated as
        /// unrepairable (mirrors `DepsFromPin`).
        pack_name: Option<String>,
    },
    /// Tenth fix (deps-from hardening series): the workspace's own
    /// two-sided conda RANGE pin (`>=X,<Y`) for `package` conflicts with a
    /// NAMED pack's own exact companion demand (`==Z`, `Z` outside the
    /// range) -- e.g. workspace `setuptools >=68,<81` vs
    /// `isaaclab-2.3x-pack 0.54.2 would require setuptools ==83.0.0`.
    /// Distinct from the plain `CondaWidenNeeded` range fallback (no named
    /// pack): doctrine says the workspace's hand-written range is real
    /// owner intent and must NOT be widened, while a NAMED pack's exact
    /// companion pin is auto-routed (derived from whatever uv's own
    /// closure happened to lock inside that pack, see `handler/mod.rs`'s
    /// `bundle.auto_routed` emission -- the only mechanism that produces
    /// an exact companion conda run-dep for a pack this backend composed)
    /// and is not a real constraint at all. Repair injects the workspace's
    /// range into the PACK's own uv closure as a pypi override instead of
    /// widening anything.
    CondaRangeVsPackPin {
        /// The conda package name both sides pin (e.g. `setuptools`).
        package: String,
        /// The workspace's own range, in conda syntax (e.g. `">=68,<81"`).
        conda_range: String,
        /// The pack's exact demanded version (e.g. `"83.0.0"`),
        /// informational (ledger old_spec).
        pack_demand: String,
        /// The pack/bundle name whose closure the override is written
        /// into (always present -- this variant only exists when a named
        /// pack was captured; see [`Conflict::DepsFromPin::pack_name`] for
        /// the `None` convention used elsewhere).
        pack_name: String,
    },
    /// Eleventh fix (deps-from hardening series, run 13): a package
    /// already repaired by `CondaRangeVsPackPin` (the pack's own
    /// auto-routed exact demand for `package` re-rendered to
    /// `pack_demand`, uv's max-in-range pick inside the injected
    /// workspace range) still fails because a DEEPER conda run-dependency
    /// of the same pack (reached transitively, e.g.
    /// `dex-retargeting -> pytorch`) imposes an additional upper-bound cap
    /// on the same `package` that the injected range doesn't respect
    /// (e.g. `pytorch` requires `setuptools <76`, nested several levels
    /// below the pack's own top-level "would require" clause). Distinct
    /// from `CondaRangeVsPackPin`: there is no NEW workspace range to
    /// derive here, only a narrower cap to intersect into the range that
    /// repair already injected -- see `conda_range_vs_pack_pin`'s
    /// ledgered override, read back via
    /// `pack_overrides::overrides_for_pack`, then intersected with this
    /// conflict's `cap_op`/`cap_version`.
    NestedCondaCap {
        /// The conda package name whose existing pack override needs
        /// narrowing (e.g. `setuptools`).
        package: String,
        /// The pack/bundle name whose ledgered override should be
        /// narrowed (e.g. `isaaclab-2.3x-pack`).
        pack_name: String,
        /// The pack's own current (auto-routed) exact demand for
        /// `package` (e.g. `"80.10.2"`), informational only.
        pack_demand: String,
        /// Comparison operator of the newly discovered nested cap
        /// (`<`/`<=`).
        cap_op: String,
        /// The nested cap's version ceiling (e.g. `"76"`).
        cap_version: String,
        /// The package whose own conda run-dependency imposed the cap
        /// (e.g. `pytorch`), informational only.
        via: String,
    },
}

impl Conflict {
    pub fn kind(&self) -> &'static str {
        match self {
            Conflict::NoCandidates { .. } => "NoCandidates",
            Conflict::CondaBoundary { .. } => "CondaBoundary",
            Conflict::PypiInternal { .. } => "PypiInternal",
            Conflict::CondaWidenNeeded { .. } => "CondaWidenNeeded",
            Conflict::DepsFromPin { .. } => "DepsFromPin",
            Conflict::NoWheelTransitive { .. } => "NoWheelTransitive",
            Conflict::CondaRangeVsPackPin { .. } => "CondaRangeVsPackPin",
            Conflict::NestedCondaCap { .. } => "NestedCondaCap",
        }
    }
}

pub trait ConflictParser {
    /// `None` = unparseable => caller rolls back according to driver policy.
    fn parse(&self, stderr: &str) -> Option<Conflict>;
}

pub struct RegexConflictParser {
    ansi: Regex,
    no_candidates: Regex,
    help_conda: Regex,
    // uv-closure JSON-RPC ErrorObject shape (pixi-build-retread wraps a uv
    // resolver failure in a JSON-RPC error; pixi's miette renderer then
    // word-wraps *that* debug-printed struct across physical lines with its
    // own `│`/`├─▶`/`╰─▶` tree gutters, on top of uv's own embedded `\n`
    // escapes). These three do the unwrap: pull the message field out,
    // strip the per-line gutter noise, then flatten to one logical line.
    uv_closure_message: Regex,
    gutter_line_prefix: Regex,
    whitespace: Regex,
    // Backend's structured hint (src/handler or src/uv_closure.rs on the
    // backend side) naming which conda package/table pinned the losing
    // range -- lets us recover a widen floor from a two-range uv pubgrub
    // disjunction without guessing which side is the conda-injected one.
    conda_provenance: Regex,
    no_conda_constraint_named: Regex,
    no_version_of: Regex,
    // Bounded requiring-chain walk: uv's pubgrub explanation prose reads
    // "Because torch==2.11.0 depends on cuda-bindings...", "And because
    // isaacsim-core==6.0.0.1 depends on torch==2.11.0, ...". Each match is
    // one (requirer, dependency) hop; `parse_uv_closure_message` walks
    // these from the conda-provenance footer package outward to find the
    // package that actually carries a user conda pin (attribution fix).
    requiring_chain: Regex,
    // Fix #20: the pack/bundle name a uv-closure conflict was raised
    // inside ("computing uv closure for bundle `X`:"), always present in
    // the backend's ErrorObject message ahead of the resolver prose.
    // Lets repair find (and write auto-overrides into) THIS pack's own
    // `.retread/overrides.json` ledger entry instead of guessing.
    uv_closure_bundle: Regex,
    // Direct (non-JSON-RPC) conda-solver prose for a resolvo-style
    // incompatible-range report (e.g. newer pixi's "cannot be installed
    // because there are no viable options" / "would require" phrasing),
    // distinct from the older "pinned by the conda solve" shape above.
    conda_incompatible: Regex,
    // Fix #21: same "cannot be installed ... no viable options" shape, but
    // for an EXACT (`==`/`=`) conda pin rather than a `<`/`<=` upper bound
    // -- the "companion" case where a fix-#20 pack override (e.g. torch)
    // makes the pack re-emit an exact `==` companion pin (torchvision,
    // torchaudio) that clashes with the workspace's own exact conda pin.
    // Kept distinct from `conda_incompatible` (rather than widening its
    // operator class in place) so the widen path's behavior/tests are
    // untouched.
    conda_incompatible_exact: Regex,
    // Run 8 (setuptools/isaaclab-2.3x-pack): same "cannot be installed ...
    // no viable options" header, but the workspace's OWN pin is a
    // two-sided RANGE (`>=68,<81`), not the bare `<`/`<=` upper bound
    // `conda_incompatible` expects -- so that regex never matches (its
    // single-operator group can't see past the leading `>=68,` prefix)
    // and the whole tree fell through to "could not parse solver error".
    // Distinct regex (rather than making the op-group of `conda_incompatible`
    // optional-prefix) so the simpler, already-covered bare shape's
    // behavior/tests stay untouched.
    conda_incompatible_range: Regex,
    // Generic (name-agnostic) half of the `DepsFromPin` shape:
    // uv's "Because P==V depends on B>=F" clause. Package `B`'s name is
    // captured but not yet anchored -- the caller re-checks it against a
    // dynamically-escaped "your project depends on B==W" regex (the
    // `regex` crate has no backreferences, so this can't be one pattern).
    deps_from_because: Regex,
    // `NoWheelTransitive` generic (name-agnostic) half: uv's "<pkg><range>
    // has no usable wheels" clause (a wildcard/range-pinned transitive dep
    // with zero wheel-bearing builds under `--no-build`), distinct from
    // the bare "Because <pkg> has no wheels" shape `uv_closure.rs`'s
    // sdist-only self-heal already recognizes (that ladder tries this
    // package FIRST; this parser branch only ever sees the message when
    // both of that ladder's rungs -- conda-route, sdist auto-build --
    // already failed for it). Package name is captured but not yet
    // anchored, same two-pass reasoning as `deps_from_because`.
    no_usable_wheels_range: Regex,
}

impl RegexConflictParser {
    pub const CALIBRATED: &'static [&'static str] = &["0.70"];

    pub fn new() -> Self {
        Self {
            ansi: Regex::new(r"\x1b\[[0-9;]*m").expect("valid ansi regex"),
            // dumb-hack.py line 47:
            // No candidates were found for ([a-zA-Z0-9_-]+)\s*==\s*([0-9][0-9a-zA-Z.]*[0-9a-zA-Z])
            no_candidates: Regex::new(
                r"No candidates were found for ([a-zA-Z0-9_-]+)\s*==\s*([0-9][0-9a-zA-Z.]*[0-9a-zA-Z])",
            )
            .expect("valid no-candidates regex"),
            // dumb-hack.py line 61, with DOTALL:
            // pinned by the conda solve.*?\n\s+([a-zA-Z0-9_-]+)==([0-9][0-9a-zA-Z._-]*)
            help_conda: Regex::new(
                r"(?s)pinned by the conda solve.*?\n\s+([a-zA-Z0-9_-]+)==([0-9][0-9a-zA-Z._-]*)",
            )
            .expect("valid help-text regex"),
            uv_closure_message: Regex::new(r#"(?s)message: "(.*?)",[\s│├╰▶─×]*data:"#)
                .expect("valid uv-closure message regex"),
            gutter_line_prefix: Regex::new(r"(?m)^[\s│├╰▶─×]+")
                .expect("valid gutter-prefix regex"),
            whitespace: Regex::new(r"\s+").expect("valid whitespace regex"),
            conda_provenance: Regex::new(
                r"package `([a-zA-Z0-9_.-]+)` is named in the conflict but conda pins `[a-zA-Z0-9_.-]+\s*(>=|>|==)\s*([0-9][0-9a-zA-Z.]*)",
            )
            .expect("valid conda-provenance regex"),
            no_conda_constraint_named: Regex::new(
                r"no generated conda constraint was named in uv's message",
            )
            .expect("valid no-constraint-named regex"),
            no_version_of: Regex::new(
                r"there is no version of ([a-zA-Z0-9_-]+)\s*==\s*([0-9][0-9a-zA-Z.]*(?:rc|a|b|dev)?[0-9a-zA-Z.]*)",
            )
            .expect("valid no-version-of regex"),
            requiring_chain: Regex::new(
                r"(?i)because ([a-zA-Z0-9_.\[\]-]+)==[0-9a-zA-Z.]+ depends on ([a-zA-Z0-9_.\[\]-]+)",
            )
            .expect("valid requiring-chain regex"),
            uv_closure_bundle: Regex::new(r"computing uv closure for bundle `([^`]+)`")
                .expect("valid uv-closure-bundle regex"),
            conda_incompatible: Regex::new(
                r"(?s)([a-zA-Z0-9_-]+)\s*(<|<=)\s*([0-9][0-9a-zA-Z.]*)[\s│├╰└─▶]+cannot be installed[\s│├╰└─▶]+because there are no viable options",
            )
            .expect("valid conda-incompatible regex"),
            // Real rattler output can word-wrap the "cannot be installed"/
            // "because there are no viable options" phrases themselves
            // mid-sentence at arbitrary terminal widths (e.g. "cannot be\n
            // installed because..."), unlike the widen-shape fixtures seen
            // so far -- tolerate whitespace/newlines between every word,
            // not just around the gutter-decorated boundary.
            conda_incompatible_exact: Regex::new(
                r"(?s)([a-zA-Z0-9_-]+)\s*(==|=)\s*([0-9][0-9a-zA-Z.]*)[\s│├╰└─▶]+cannot\s+be\s+installed[\s│├╰└─▶]+because\s+there\s+are\s+no\s+viable\s+options",
            )
            .expect("valid conda-incompatible-exact regex"),
            // Tenth fix (deps-from hardening): now captures the RANGE's
            // floor op/value too (groups 2-3), not just the ceiling
            // (groups 4-5) -- needed to reconstruct the full workspace
            // range (`>=68,<81`) for `CondaRangeVsPackPin`'s repair
            // instead of just the ceiling half the old widen-only path
            // used.
            conda_incompatible_range: Regex::new(
                r"(?s)([a-zA-Z0-9_-]+)\s*(>=|>)\s*([0-9][0-9a-zA-Z.]*)\s*,\s*(<|<=)\s*([0-9][0-9a-zA-Z.]*)[\s│├╰└─▶]+cannot\s+be\s+installed[\s│├╰└─▶]+because\s+there\s+are\s+no\s+viable\s+options",
            )
            .expect("valid conda-incompatible-range regex"),
            deps_from_because: Regex::new(
                r"(?i)because\s+([a-zA-Z0-9_.\[\]-]+)==[0-9a-zA-Z.]+\s+depends\s+on\s+([a-zA-Z0-9_.\[\]-]+)(?:\[[^\]]*\])?\s*(>=|>)\s*([0-9][0-9a-zA-Z.]*)",
            )
            .expect("valid deps-from-because regex"),
            no_usable_wheels_range: Regex::new(
                r"(?i)([a-zA-Z0-9][a-zA-Z0-9._-]*)((?:>=|<=|==|>|<)\S*)\s+has no usable wheels",
            )
            .expect("valid no-usable-wheels-range regex"),
        }
    }

    /// Pulls the uv resolver's message out of a pixi-build-retread
    /// JSON-RPC `ErrorObject { .. }` and flattens miette's line-wrapped,
    /// gutter-decorated rendering (plus uv's own embedded `\n` escapes)
    /// back into a single logical line, so the regexes below can match
    /// across what were originally several physical/escaped lines.
    fn extract_uv_closure_message(&self, stderr: &str) -> Option<String> {
        let caps = self.uv_closure_message.captures(stderr)?;
        let raw = &caps[1];
        // Rejoin gutter-stripped physical lines with a space -- UNLESS the
        // previous line ends in a hyphen, in which case miette's own
        // textwrap broke a hyphenated identifier mid-word (real observed
        // shape: a long name like `antlr4-python3-runtime` line-wraps as
        // `antlr4-python3-` / `runtime...` with no whitespace at the
        // break) and the fragments must rejoin with NO space so a whole-
        // name regex still sees one contiguous token. Every other wrap is
        // a genuine word boundary and keeps its separating space, exactly
        // as before.
        let mut joined = String::new();
        for line in raw.lines() {
            let stripped = self.gutter_line_prefix.replace(line, "");
            if !joined.is_empty() && !joined.ends_with('-') {
                joined.push(' ');
            }
            joined.push_str(&stripped);
        }
        let unescaped = joined.replace("\\n", " ");
        let collapsed = self.whitespace.replace_all(&unescaped, " ");
        Some(collapsed.trim().to_string())
    }

    /// Classifies an already-flattened uv-closure message into a
    /// `Conflict`, or `None` if it doesn't match a known shape (e.g. the
    /// backend's other JSON-RPC errors that aren't resolver conflicts at
    /// all, like "package X has no wheels in the exported closure").
    fn parse_uv_closure_message(&self, msg: &str) -> Option<Conflict> {
        let pack_name = self
            .uv_closure_bundle
            .captures(msg)
            .map(|c| c[1].to_string());
        if let Some(caps) = self.conda_provenance.captures(msg) {
            let package = caps[1].to_string();
            let conda_op = caps[2].to_string();
            let conda_floor = caps[3].to_string();
            let escaped = regex::escape(&package);
            let dual_range = Regex::new(&format!(
                r"depends on {escaped}(?:\{{[^}}]*\}})?\s*(>=|>)\s*([0-9][0-9a-zA-Z.]*),<[0-9a-zA-Z.]+\s+and\s+{escaped}(?:\{{[^}}]*\}})?\s*(>=|>)\s*([0-9][0-9a-zA-Z.]*),<[0-9a-zA-Z.]+"
            ))
            .ok()?;
            let dcaps = dual_range.captures(msg)?;
            let (op1, floor1) = (dcaps[1].to_string(), dcaps[2].to_string());
            let (op2, floor2) = (dcaps[3].to_string(), dcaps[4].to_string());
            let (op, floor) = if op1 == conda_op && floor1 == conda_floor {
                (op2, floor2)
            } else {
                (op1, floor1)
            };
            let requiring_chain = self.extract_requiring_chain(msg, &package);
            return Some(Conflict::CondaWidenNeeded {
                package,
                op,
                floor,
                conda_version: format!("{conda_op}{conda_floor}"),
                requiring_chain,
                pack_name,
            });
        }

        if self.no_conda_constraint_named.is_match(msg)
            && let Some(caps) = self.no_version_of.captures(msg)
        {
            return Some(Conflict::PypiInternal {
                package: caps[1].to_string(),
                version: caps[2].to_string(),
            });
        }

        // Intrinsic pypi-vs-pypi conflict (no conda pin named at all --
        // same `no_conda_constraint_named` footer as the `PypiInternal`
        // shape above, but a different pubgrub prose shape: uv's own
        // "Because P==V depends on B>=F and your project depends on
        // B==W" two-clause explanation, where B's `==W` pin is a
        // `retread-deps-from` root, not a hand-authored one). Two-pass
        // match (generic "because" capture, then a name-anchored second
        // regex) since the `regex` crate has no backreferences.
        if self.no_conda_constraint_named.is_match(msg)
            && let Some(caps) = self.deps_from_because.captures(msg)
        {
            let requirer = caps[1].to_string();
            let package = caps[2].to_string();
            let op = caps[3].to_string();
            let floor = caps[4].to_string();
            let escaped = regex::escape(&package);
            if let Ok(pin_re) = Regex::new(&format!(
                r"(?i)your project depends on {escaped}(?:\[[^\]]*\])?\s*(==|===)\s*([0-9][0-9a-zA-Z.]*)"
            )) && let Some(pcaps) = pin_re.captures(msg)
            {
                return Some(Conflict::DepsFromPin {
                    package,
                    pinned_version: pcaps[2].to_string(),
                    op,
                    floor,
                    requirer,
                    pack_name,
                });
            }
        }

        // `NoWheelTransitive`: same `no_conda_constraint_named` footer, a
        // THIRD pubgrub prose shape -- "<pkg><range> has no usable
        // wheels ... And because <requirer>==<requirer_pin> depends on
        // <pkg>==<wildcard>". Only reached when the backend's sdist-only
        // self-heal ladder already exhausted the conda-route and
        // sdist-auto-build rungs for `pkg` (see `uv_closure.rs`); by the
        // time it surfaces here there is no route left except relaxing
        // the requirer's own exact pin. Two-pass match, same reasoning as
        // `deps_from_because` above.
        if self.no_conda_constraint_named.is_match(msg)
            && let Some(caps) = self.no_usable_wheels_range.captures(msg)
        {
            let package = caps[1].to_string();
            let range = caps[2].to_string();
            let escaped = regex::escape(&package);
            if let Ok(requirer_re) = Regex::new(&format!(
                r"(?i)because\s+([a-zA-Z0-9_.\[\]-]+)==([0-9][0-9a-zA-Z.]*)\s+depends\s+on\s+{escaped}(?:\[[^\]]*\])?\s*(==|===)\s*[0-9a-zA-Z.*]+"
            )) && let Some(rcaps) = requirer_re.captures(msg)
            {
                return Some(Conflict::NoWheelTransitive {
                    package,
                    range,
                    requirer: rcaps[1].to_string(),
                    requirer_pin: rcaps[2].to_string(),
                    pack_name,
                });
            }
        }

        None
    }

    /// Bounded breadth-first walk of uv's "Because X==V depends on Y"
    /// pubgrub prose, starting from the conda-provenance footer package and
    /// following every hop to *each* package that named it as a
    /// dependency (not just the first match), nearest first, then their
    /// requirers in turn. Stops after `MAX_HOPS` layers or on a repeated
    /// name (cycle guard). Used to re-attribute a `CondaWidenNeeded`
    /// conflict when the footer package (the transitive symptom, e.g.
    /// `cuda-bindings`) has no user conda pin of its own -- the real fix
    /// belongs on whichever ancestor (e.g. `torch`) the user *does* pin
    /// (directly or via the conda name map, e.g. `pytorch-gpu`).
    ///
    /// A single-path walk (stop at the first requirer found) misses
    /// SIBLING requirers of the same footer/hop -- e.g. when both `torch`
    /// and `torchvision` independently depend on `cuda-bindings`, a
    /// strict walk finds only `torch` and, once that's already in
    /// tried-state (fix #19: torch overridden in a prior iteration), the
    /// caller has nowhere else to look even though `torchvision` (right
    /// there in the same report, with its own user conda pin) would work.
    /// Collecting every requirer at each layer, in report order, and
    /// letting the caller skip already-tried entries fixes that: "first
    /// untried wins" over ALL pin-owning candidates, not just the nearest.
    fn extract_requiring_chain(&self, msg: &str, footer_package: &str) -> Vec<String> {
        const MAX_HOPS: usize = 6;
        let pairs: Vec<(String, String)> = self
            .requiring_chain
            .captures_iter(msg)
            .map(|c| (c[1].to_string(), c[2].to_string()))
            .collect();
        let mut chain = Vec::new();
        let mut frontier = vec![footer_package.to_string()];
        for _ in 0..MAX_HOPS {
            let mut next_frontier = Vec::new();
            for current in &frontier {
                for (requirer, dep) in &pairs {
                    if !dep.eq_ignore_ascii_case(current) {
                        continue;
                    }
                    if requirer == footer_package || chain.iter().any(|p: &String| p == requirer) {
                        continue;
                    }
                    chain.push(requirer.clone());
                    next_frontier.push(requirer.clone());
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        chain
    }

    /// Direct (non-JSON-RPC) resolvo-style "cannot be installed because
    /// there are no viable options" / "would require" prose -- a second
    /// conflict shape distinct from the older `help_conda` "pinned by the
    /// conda solve" text, seen on newer pixi versions.
    fn parse_conda_incompatible(&self, stderr: &str) -> Option<Conflict> {
        // Widen shape first (pinned `<`/`<=`, pypi side wants a `>=`/`>`
        // floor) -- unchanged from before fix #21, no pack-name attribution
        // (this class resolves via the workspace conda-pin-owner path, not
        // the pack-override path; see `conda_widen_needed`).
        if let Some(caps) = self.conda_incompatible.captures(stderr) {
            let package = caps[1].to_string();
            let conda_op = caps[2].to_string();
            let conda_version = caps[3].to_string();
            let escaped = regex::escape(&package);
            // Between "would require" and the package name, resolvo's tree
            // rendering inserts its own `└─`-style gutter line (in addition
            // to any outer miette gutter); tolerate box-drawing chars as
            // well as whitespace in that gap.
            let would_require = Regex::new(&format!(
                r"(?s)would require[\s│├╰└─▶]*{escaped}\s*(>=|>)\s*([0-9][0-9a-zA-Z.]*)"
            ))
            .ok()?;
            if let Some(rcaps) = would_require.captures(stderr) {
                return Some(Conflict::CondaWidenNeeded {
                    package,
                    op: rcaps[1].to_string(),
                    floor: rcaps[2].to_string(),
                    conda_version: format!("{conda_op}{conda_version}"),
                    requiring_chain: Vec::new(),
                    pack_name: None,
                });
            }
        }

        // Fix #21: exact-pin companion shape (pinned `==`/`=`, the pack's
        // own re-emitted companion also demands an exact `==`/`=`). The
        // workspace pin is authoritative here (conda-as-truth), so the
        // fix belongs on the PACK's side -- carry `pack_name` (parsed from
        // the "<pack> X.Y.Z would require" clause immediately preceding
        // the package's requiring clause; the pack-version prefix is what
        // distinguishes this from the OTHER "<ver> would require" clauses
        // in the same prose, e.g. the unrelated `python_abi` requirement)
        // so the existing fix-#20 `resolve_pack_override` path fires
        // instead of the workspace conda-pin-owner scan.
        if let Some(caps) = self.conda_incompatible_exact.captures(stderr) {
            let package = caps[1].to_string();
            let conda_op = caps[2].to_string();
            let conda_version = caps[3].to_string();
            let escaped = regex::escape(&package);
            let would_require_exact = Regex::new(&format!(
                r"(?s)([a-zA-Z][a-zA-Z0-9_.-]*)\s+[0-9][0-9a-zA-Z.]*\s+would require[\s│├╰└─▶]*{escaped}\s*(==|=)\s*([0-9][0-9a-zA-Z.]*)"
            ))
            .ok()?;
            if let Some(rcaps) = would_require_exact.captures(stderr) {
                return Some(Conflict::CondaWidenNeeded {
                    package,
                    op: rcaps[2].to_string(),
                    floor: rcaps[3].to_string(),
                    conda_version: format!("{conda_op}{conda_version}"),
                    requiring_chain: Vec::new(),
                    pack_name: Some(rcaps[1].to_string()),
                });
            }
        }

        // Run 8: the workspace's own conda pin is a `>=X,<Y` RANGE and a
        // sibling pack's stub demands an exact version above the range's
        // ceiling (e.g. workspace `setuptools >=68,<81` vs
        // `isaaclab-2.3x-pack 0.54.2 would require setuptools ==83.0.0`).
        // Doctrine mirrors the plain (bare `<`/`<=`) widen shape above, NOT
        // the exact-companion shape: the workspace range is a manual cap
        // the user chose for an unrelated reason (e.g. a different pack's
        // `--editable` requirement), not a hard anchor, so conda-as-truth
        // says WIDEN the workspace's own pin rather than override the
        // pack (no `pack_name` -- routes through the workspace
        // conda-pin-owner scan in `conda_widen_needed`, same as
        // `conda_incompatible`). The pack's `==`/`=` demand is normalized
        // to a `>=` floor (not re-emitted as an exact pin): conda-as-truth
        // only asserts a MINIMUM version is required, matching the
        // `deps-from` exact-pin-softening precedent elsewhere in this
        // ladder.
        if let Some(caps) = self.conda_incompatible_range.captures(stderr) {
            let package = caps[1].to_string();
            let floor_op = caps[2].to_string();
            let floor = caps[3].to_string();
            let ceil_op = caps[4].to_string();
            let ceil = caps[5].to_string();
            let escaped = regex::escape(&package);

            // Tenth fix (deps-from hardening): if a NAMED pack's own
            // render demands an EXACT companion version (the auto-routed
            // shape `handler/mod.rs`'s `bundle.auto_routed` emits), route
            // to `CondaRangeVsPackPin` -- the workspace range wins, the
            // pack's exact pin is what gets relaxed, via a pypi override
            // written into THAT pack's own closure. Checked before the
            // name-agnostic fallback below so a named pack + exact demand
            // always takes the conda-as-truth direction (never widens the
            // workspace's own hand-written range).
            let would_require_pack_exact = Regex::new(&format!(
                r"(?s)([a-zA-Z][a-zA-Z0-9_.-]*)\s+[0-9][0-9a-zA-Z.]*\s+would require[\s│├╰└─▶]*{escaped}\s*(==|=)\s*([0-9][0-9a-zA-Z.]*)"
            ))
            .ok()?;
            if let Some(rcaps) = would_require_pack_exact.captures(stderr) {
                return Some(Conflict::CondaRangeVsPackPin {
                    package,
                    conda_range: format!("{floor_op}{floor},{ceil_op}{ceil}"),
                    pack_demand: rcaps[3].to_string(),
                    pack_name: rcaps[1].to_string(),
                });
            }

            // Fallback (no named pack captured, or the demand wasn't a
            // bare exact version): unchanged from before this fix --
            // widen the workspace's own pin, since there is no pack to
            // attribute an override to.
            let would_require = Regex::new(&format!(
                r"(?s)would require[\s│├╰└─▶]*{escaped}\s*(==|=|>=|>)\s*([0-9][0-9a-zA-Z.]*)"
            ))
            .ok()?;
            if let Some(rcaps) = would_require.captures(stderr) {
                return Some(Conflict::CondaWidenNeeded {
                    package,
                    op: ">=".to_string(),
                    floor: rcaps[2].to_string(),
                    conda_version: format!("{ceil_op}{ceil}"),
                    requiring_chain: Vec::new(),
                    pack_name: None,
                });
            }
        }

        self.parse_nested_conda_cap(stderr)
    }

    /// Run-13 shape (eleventh fix): a NESTED resolvo tree where a named
    /// pack's own top-level "would require {package} =={V}" clause (the
    /// auto-routed pin a prior `CondaRangeVsPackPin` repair produced) is
    /// followed, several tree levels further down the SAME error, by an
    /// unrelated conda run-dep's own "would require {package} (<|<=){V2}"
    /// clause -- e.g.:
    ///
    /// ```text
    /// isaaclab-2.3x-pack 0.54.2 would require
    ///   setuptools ==80.10.2, which cannot be installed ...
    ///   dex-retargeting >=0.4.6,<0.5, which cannot be installed ...
    ///     dex-retargeting 0.4.6 would require
    ///       pytorch *, which cannot be installed ...
    ///         pytorch 2.7.0 | ... would require
    ///           setuptools <76, which cannot be installed ...
    /// ```
    ///
    /// Neither `conda_incompatible_exact` nor `conda_incompatible` match
    /// this text: both require "cannot be installed" to follow the
    /// package/version with only gutter/whitespace in between, but this
    /// shape always has a ", which" aside in the way -- hence run 13's
    /// EXIT=2 (unparseable). This is a distinct, narrower parse pass (not
    /// a fix to those regexes' char classes) since widening them to
    /// tolerate ", which ..." would also change behavior for every
    /// already-covered shape.
    fn parse_nested_conda_cap(&self, stderr: &str) -> Option<Conflict> {
        // Outer clause: "<pack> <ver> would require <package> ==<val>",
        // same shape `conda_incompatible_exact`'s pack-exact branch looks
        // for, but WITHOUT requiring "cannot be installed" to immediately
        // follow (this shape's own "cannot be installed" is separated by
        // the ", which" aside those other regexes don't tolerate).
        let outer = Regex::new(
            r"(?s)([a-zA-Z][a-zA-Z0-9_.-]*)\s+[0-9][0-9a-zA-Z.]*\s+would\s+require[\s│├╰└─▶]*([a-zA-Z0-9_-]+)\s*(?:==|=)\s*([0-9][0-9a-zA-Z.]*)",
        )
        .ok()?;
        let outer_caps = outer.captures(stderr)?;
        let pack_name = outer_caps[1].to_string();
        let package = outer_caps[2].to_string();
        let pack_demand = outer_caps[3].to_string();
        let outer_end = outer_caps.get(0)?.end();

        // Nested clause, searched only in the text AFTER the outer match:
        // "<via> <ver>[ | <ver> ...] would require <package> (<|<=)<val>".
        // The version-list char class deliberately excludes letters other
        // than the digits/dots/pipes a resolvo version disjunction uses,
        // so it can't accidentally swallow the "would require" keywords
        // themselves and match some unrelated, much later clause.
        let escaped = regex::escape(&package);
        let nested = Regex::new(&format!(
            r"(?s)([a-zA-Z][a-zA-Z0-9_.-]*)\s+[0-9.]+(?:\s*\|\s*[0-9.]+)*\s+would\s+require[\s│├╰└─▶]*{escaped}\s*(<=|<)\s*([0-9][0-9a-zA-Z.]*)"
        ))
        .ok()?;
        let nested_caps = nested.captures(&stderr[outer_end..])?;

        Some(Conflict::NestedCondaCap {
            package,
            pack_name,
            pack_demand,
            cap_op: nested_caps[2].to_string(),
            cap_version: nested_caps[3].to_string(),
            via: nested_caps[1].to_string(),
        })
    }

    pub fn strip_ansi<'a>(&self, stderr: &'a str) -> std::borrow::Cow<'a, str> {
        self.ansi.replace_all(stderr, "")
    }

    /// Generic ownership-driven fallback engine (ends the
    /// rung-per-error-shape treadmill): flattens `stderr` the same way
    /// [`Self::extract_uv_closure_message`] does for a backend uv-closure
    /// JSON-RPC error, falling back to a plain ANSI-strip + gutter-strip +
    /// whitespace-collapse for direct (non-JSON-RPC) resolvo/conda-solver
    /// prose -- either way producing one logical line the mention regex
    /// can scan without per-shape special-casing.
    fn flatten_for_generic_scan(&self, stderr: &str) -> String {
        if let Some(msg) = self.extract_uv_closure_message(stderr) {
            return msg;
        }
        let stripped = self.ansi.replace_all(stderr, "");
        let mut joined = String::new();
        for line in stripped.lines() {
            let cleaned = self.gutter_line_prefix.replace(line, "");
            if !joined.is_empty() && !joined.ends_with('-') {
                joined.push(' ');
            }
            joined.push_str(&cleaned);
        }
        self.whitespace.replace_all(&joined, " ").trim().to_string()
    }

    /// Generic (shape-agnostic) extraction of every
    /// `<package-name> <version-specifier>` clause anywhere in a failed
    /// solve's error text -- the corpus in `tests/fixtures/solve_errors/`
    /// covers uv-closure JSON-RPC errors, direct rattler/resolvo prose,
    /// and the older "pinned by the conda solve" shape alike, and this one
    /// permissive regex family finds the relevant package(s) in all of
    /// them, whatever the surrounding prose ("no candidates were found
    /// for", "would require", "depends on", "is needed", ...). Handles
    /// `==X`, `>=X`, `<X`, `<=X`, `>X`, a single trailing `,<Y`/`,<=Y`
    /// second clause (the common two-sided range shape), and a trailing
    /// glob (`X.Y.*`) verbatim. This is the FALLBACK extractor -- existing
    /// per-shape parsers in [`RegexConflictParser::parse`] still run
    /// first; this only ever needs to fire when none of them matched.
    pub fn extract_generic_mentions(&self, stderr: &str) -> Vec<Mention> {
        let text = self.flatten_for_generic_scan(stderr);
        // A package/spec mention: a name, an operator, a version (with an
        // optional trailing `.*` glob), and an optional second
        // `,<op><version>` clause completing a two-sided range.
        let mention_re = Regex::new(
            r"(?i)\b([A-Za-z][A-Za-z0-9._-]{1,60}?)\s*(>=|<=|==|=|>|<)\s*([0-9][0-9A-Za-z.]*\*?)(?:\s*,\s*(>=|<=|==|=|>|<)\s*([0-9][0-9A-Za-z.]*\*?))?",
        )
        .expect("valid generic mention regex");
        // Best-effort requirer attribution: the nearest PRECEDING "<name>
        // [==<ver>] would require" / "because <name>==<ver> depends on"
        // clause, whichever package it actually turns out to name --
        // informational only (`Mention::requirer` is `Option`), never
        // load-bearing for the extraction itself.
        let requirer_re = Regex::new(
            r"(?i)\b([A-Za-z][A-Za-z0-9._-]*)\s*(?:==|=)?\s*[0-9][0-9A-Za-z.]*(?:\s*\|\s*[0-9A-Za-z.]+)*\s+(?:would\s+require|depends\s+on)",
        )
        .expect("valid generic requirer regex");
        let requirer_positions: Vec<(usize, String)> = requirer_re
            .captures_iter(&text)
            .map(|c| {
                (
                    c.get(0).expect("group 0 always matches").start(),
                    c[1].to_string(),
                )
            })
            .collect();

        let mut mentions = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for caps in mention_re.captures_iter(&text) {
            let package = caps[1].to_string();
            if package.len() < 2 {
                continue;
            }
            fn norm_op(op: &str) -> &str {
                if op == "=" { "==" } else { op }
            }
            let op1 = norm_op(&caps[2]);
            let v1 = &caps[3];
            let spec = match (caps.get(4), caps.get(5)) {
                (Some(op2), Some(v2)) => {
                    format!("{op1}{v1},{}{}", norm_op(op2.as_str()), v2.as_str())
                }
                _ => format!("{op1}{v1}"),
            };
            let start = caps.get(0).expect("group 0 always matches").start();
            let requirer = requirer_positions
                .iter()
                .filter(|(pos, name)| *pos < start && !name.eq_ignore_ascii_case(&package))
                .max_by_key(|(pos, _)| *pos)
                .map(|(_, name)| name.clone());
            let key = (package.clone(), spec.clone());
            if seen.insert(key) {
                mentions.push(Mention {
                    package,
                    spec,
                    requirer,
                });
            }
        }
        mentions
    }

    /// The pack/bundle name a conflict was raised inside (`computing uv
    /// closure for bundle `<name>`:`), when present -- same signature
    /// [`Self::parse_uv_closure_message`] already reads via
    /// `uv_closure_bundle`, exposed standalone so the generic fallback
    /// engine can scope its ownership classification the same way the
    /// specific parsers do, even though it never calls
    /// `parse_uv_closure_message` itself.
    pub fn extract_bundle_name(&self, stderr: &str) -> Option<String> {
        if let Some(name) = self
            .uv_closure_bundle
            .captures(stderr)
            .map(|c| c[1].to_string())
        {
            return Some(name);
        }
        // Fallback for direct (non-JSON-RPC) resolvo/conda-solver prose,
        // which never carries a "computing uv closure for bundle `X`:"
        // label at all: a "<name> <ver> would require" clause's subject
        // is, in every rung this hardening series has seen, the
        // pack/bundle this workspace composed (the same capture
        // `conda_incompatible_exact`/`conda_range_vs_pack_pin`'s regexes
        // already rely on) -- best-effort, only actually load-bearing
        // downstream once `resolve_pack_dir` confirms it names one of
        // this workspace's own composed packs.
        //
        // When resolvo names MULTIPLE packs (run-17's pandas fixture:
        // both `isaaclab-2.3x-pack` and `protomotions-deps-pack` appear,
        // each with their own single-version "would require" clause),
        // the textually-FIRST match is not necessarily the actual
        // failing pack -- resolvo prints a pack's SATISFIABLE branch
        // ("* can be installed with any of the following options") for
        // context before the branch that actually dead-ends ("* cannot
        // be installed because there are no viable options"). Prefer a
        // "would require" match whose nearest preceding branch marker is
        // "cannot be installed" (a real dead end) over one under "can be
        // installed" (a satisfiable alternative merely shown for
        // context); among equally-qualified matches, keep the first
        // (outermost), preserving prior behavior for single-pack
        // fixtures and fixtures with no branch markers at all.
        //
        // Terminal-wrapped source text (every fixture decoded verbatim
        // from a captured log) routinely breaks "cannot be installed"
        // across a line boundary mid-phrase ("cannot be\n    installed
        // because..."); scanning raw `stderr` for that substring misses
        // it. Flatten first (same ANSI-strip + gutter-strip +
        // whitespace-collapse `extract_generic_mentions` already relies
        // on) so both the "would require" regex and the marker substring
        // search operate on the same rewrapped text.
        let text = self.flatten_for_generic_scan(stderr);
        let would_require =
            Regex::new(r"(?s)([a-zA-Z][a-zA-Z0-9_.-]*)\s+[0-9][0-9a-zA-Z.]*\s+would\s+require")
                .ok()?;
        let mut first: Option<String> = None;
        for caps in would_require.captures_iter(&text) {
            let start = caps.get(0).expect("group 0 always matches").start();
            let name = caps[1].to_string();
            if first.is_none() {
                first = Some(name.clone());
            }
            let preceding = &text[..start];
            let cannot_pos = preceding.rfind("cannot be installed");
            let can_pos = preceding.rfind("can be installed");
            let is_dead_end_branch = match (cannot_pos, can_pos) {
                (Some(c), Some(k)) => c > k,
                (Some(_), None) => true,
                _ => false,
            };
            if is_dead_end_branch {
                return Some(name);
            }
        }
        first
    }

    pub fn is_post_widen_conda_unsat(stderr: &str) -> bool {
        stderr.contains("Cannot solve the request because of")
            || stderr.contains("The following packages are incompatible")
    }
}

/// Run-38 shape: a workspace EXACT pin dead-ends on a wrong-python
/// `python_abi` build tail --
///
/// ```text
/// torchaudio ==2.7.0 cannot be installed because there are no viable options:
/// └─ torchaudio 2.7.0 | 2.7.0 | ... would require
///    └─ python_abi 3.10.* *_cp310, for which no candidates were found.
/// ```
///
/// Resolvo elides the constraints that EXCLUDED every env-python-matching
/// build (typically another pack's emitted torch-family range, run 34/38:
/// `pytorch >=2.10.0,<3` excluding all pytorch-2.7-built cp311
/// torchaudios), so there is nothing in the tree for the repair ladder to
/// act on -- the workspace pin itself is untouchable doctrine and the real
/// excluder is unnamed. This diagnostic turns the bare "could not parse"
/// into an actionable message naming the pin and the cause class. Returns
/// `None` when the text doesn't match the shape.
pub fn diagnose_abi_build_tail(stderr: &str) -> Option<String> {
    let re = Regex::new(
        r"(?s)([A-Za-z][A-Za-z0-9._-]*)\s*==\s*([0-9][0-9A-Za-z.]*)\s+cannot be\s+installed because there are no viable options:.{0,400}?would require[\s│├╰└─▶]*python_abi\s+([0-9.]+)\.\*\s+\*_cp(\d+),\s*for which no\s+candidates were found",
    )
    .ok()?;
    let caps = re.captures(stderr)?;
    Some(format!(
        "retread lock: diagnosis -- the workspace pin `{pkg} =={ver}` only \
         reaches builds for python {abi} (cp{cp}); every build matching the \
         env's python was excluded by constraints resolvo does not show \
         (typically a sibling pack's emitted torch-family range at a \
         different version, e.g. run 34/38's `pytorch >=2.10.0,<3`). The \
         workspace pin is hand-written truth and is not repaired; the fix \
         is route-time (the pack's uv closure must be constrained to the \
         consuming envs' hand-written versions -- check the pack renders' \
         emitted ranges for the {pkg} family).",
        pkg = &caps[1],
        ver = &caps[2],
        abi = &caps[3],
        cp = &caps[4],
    ))
}

/// Run-40 companion shape to [`diagnose_abi_build_tail`]: an ABI-anchor
/// exact pin (`cuda-version ==12.8`) dead-ends on its own
/// `run_constrained` of a RENAMED/DISCONTINUED companion package --
///
/// ```text
/// cuda-version ==12.8 cannot be installed because there are no viable options:
/// └─ cuda-version 12.8 would constrain
///    └─ cudatoolkit ==12.8|12.8.*, which conflicts with any installable
///       versions previously reported
/// ```
///
/// `cudatoolkit` was renamed to the `cuda-*` split before 12.x, so no
/// 12.8 build exists: some package's SELECTED build variant is an OLD one
/// depending on `cudatoolkit` (its newer `cuda-version`-depending builds
/// were excluded by other constraints earlier in the same tree -- the
/// run-38 build-variant-exclusion pattern again). The anchor is
/// immutable; the excluding constraint (visible in the branches above the
/// clause) is what yields.
pub fn diagnose_constrained_renamed_companion(stderr: &str) -> Option<String> {
    let re = Regex::new(
        r"(?s)([A-Za-z][A-Za-z0-9._-]*)\s*==\s*([0-9][0-9A-Za-z.]*)\s+cannot be\s+installed because there are no viable\s+options:.{0,200}?would constrain[\s│├╰└─▶]*([A-Za-z][A-Za-z0-9._-]*)\s*==\s*([0-9][0-9A-Za-z.]*)\|[0-9][0-9A-Za-z.]*\.\*,\s*which conflicts with any\s+installable\s+versions previously reported",
    )
    .ok()?;
    let caps = re.captures(stderr)?;
    Some(format!(
        "retread lock: diagnosis -- the ABI anchor `{anchor} =={ver}` \
         run_constrains `{companion}` to {ver}, but no such {companion} \
         build exists (renamed/discontinued line). Some package's SELECTED \
         build variant is an old one depending on `{companion}` because its \
         newer `{anchor}`-based builds were excluded by other constraints \
         earlier in this same tree (build-variant exclusion, run 38/40 \
         pattern). The anchor is immutable; the excluding constraint in the \
         branches above this clause is what must yield (widen/un-route the \
         owning pack's emission).",
        anchor = &caps[1],
        ver = &caps[2],
        companion = &caps[3],
    ))
}

impl Default for RegexConflictParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictParser for RegexConflictParser {
    fn parse(&self, stderr: &str) -> Option<Conflict> {
        // Newest, most specific shapes first: a backend-wrapped uv-closure
        // JSON-RPC ErrorObject, or (failing that) the plain resolvo-style
        // incompatible-range prose. Both are distinct from the classic
        // conda-solver shapes handled below and are checked first since
        // they carry unambiguous signatures (`ErrorObject { .. }` /
        // "cannot be installed because there are no viable options").
        if let Some(msg) = self.extract_uv_closure_message(stderr) {
            if let Some(conflict) = self.parse_uv_closure_message(&msg) {
                return Some(conflict);
            }
            // It was a uv-closure ErrorObject, but not a resolver conflict
            // shape we recognize (e.g. "has no wheels in the exported
            // closure") -- unparseable, don't fall through to conda-prose
            // regexes that can't match this text anyway.
            return None;
        }
        if let Some(conflict) = self.parse_conda_incompatible(stderr) {
            return Some(conflict);
        }

        if let Some(caps) = self.no_candidates.captures(stderr) {
            return Some(Conflict::NoCandidates {
                package: caps[1].to_string(),
                version: caps[2].to_string(),
            });
        }

        let help = self.help_conda.captures(stderr)?;
        let package = help[1].to_string();
        let conda_version = help[2].to_string();
        let escaped = regex::escape(&package);

        let range_demand = Regex::new(&format!(
            r"depends on {escaped}\s*(>=|>)\s*([0-9][0-9a-zA-Z._-]*)"
        ))
        .ok()?;
        if let Some(caps) = range_demand.captures(stderr) {
            return Some(Conflict::CondaWidenNeeded {
                package,
                op: caps[1].to_string(),
                floor: caps[2].to_string(),
                conda_version,
                requiring_chain: Vec::new(),
                pack_name: None,
            });
        }

        let range_needed = Regex::new(&format!(
            r"{escaped}\s*(>=|>)\s*([0-9][0-9a-zA-Z._-]*)\s+is needed"
        ))
        .ok()?;
        if let Some(caps) = range_needed.captures(stderr) {
            return Some(Conflict::CondaWidenNeeded {
                package,
                op: caps[1].to_string(),
                floor: caps[2].to_string(),
                conda_version,
                requiring_chain: Vec::new(),
                pack_name: None,
            });
        }

        // dumb-hack.py line 71:
        // depends on {p}==([0-9][0-9a-zA-Z._-]*) and {p}==([0-9][0-9a-zA-Z._-]*)
        let dual = Regex::new(&format!(
            r"depends on {escaped}==([0-9][0-9a-zA-Z._-]*) and {escaped}==([0-9][0-9a-zA-Z._-]*)"
        ))
        .ok()?;
        if let Some(caps) = dual.captures(stderr) {
            let a = caps[1].to_string();
            let b = caps[2].to_string();
            let wanted = if conda_version == b {
                a
            } else if conda_version == a {
                b
            } else {
                a
            };
            return Some(Conflict::CondaBoundary {
                package,
                version: wanted,
            });
        }

        if stderr.contains("we know from (1)") {
            return Some(Conflict::PypiInternal {
                package,
                version: conda_version,
            });
        }

        Some(Conflict::CondaBoundary {
            package,
            version: conda_version,
        })
    }
}

pub fn tail(s: &str, max_chars: usize) -> String {
    let len = s.chars().count();
    if len <= max_chars {
        return s.to_string();
    }
    s.chars().skip(len - max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONDA_ABI_BUILD_TAIL_RUN38: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_abi_build_tail_torchaudio_run38.txt");

    #[test]
    fn diagnose_abi_build_tail_names_pin_and_abi_on_run38_fixture() {
        let diag = diagnose_abi_build_tail(CONDA_ABI_BUILD_TAIL_RUN38)
            .expect("run-38 build-tail shape must be recognized");
        assert!(diag.contains("torchaudio =="), "{diag}");
        assert!(diag.contains("2.7.0"), "{diag}");
        assert!(diag.contains("python 3.10"), "{diag}");
        assert!(diag.contains("cp310"), "{diag}");
        // Unrelated conflict prose must NOT match (no python_abi tail).
        assert!(
            diagnose_abi_build_tail(
                "wandb==0.23.0 depends on sentry-sdk>=2.0.0 and your project \
                 depends on sentry-sdk==1.38.0"
            )
            .is_none()
        );
    }

    const CONDA_LEDGERED_RANGE_RUN40: &str = include_str!(
        "../../tests/fixtures/solve_errors/conda_ledgered_range_top_pick_dead_end_run40.txt"
    );

    #[test]
    fn diagnose_constrained_renamed_companion_names_anchor_on_run40_fixture() {
        let diag = diagnose_constrained_renamed_companion(CONDA_LEDGERED_RANGE_RUN40)
            .expect("run-40 cudatoolkit constrain shape must be recognized");
        assert!(diag.contains("cuda-version =="), "{diag}");
        assert!(diag.contains("12.8"), "{diag}");
        assert!(diag.contains("cudatoolkit"), "{diag}");
        assert!(diagnose_constrained_renamed_companion("unrelated solver prose").is_none());
    }

    const CONDA_BOUNDARY_SINGLE_LINE: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_boundary_single_line.txt");
    const CONDA_BOUNDARY_HELP_ONLY: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_boundary_help_only.txt");
    const PYPI_INTERNAL: &str = include_str!("../../tests/fixtures/solve_errors/pypi_internal.txt");
    const NO_CANDIDATES: &str =
        include_str!("../../tests/fixtures/solve_errors/no_candidates_conda.txt");
    const NO_CANDIDATES_PYPI: &str =
        include_str!("../../tests/fixtures/solve_errors/no_candidates_pypi.txt");
    const CONDA_BOUNDARY_RANGE: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_boundary_range.txt");
    const CONDA_BOUNDARY_RANGE_ALT: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_boundary_range_alt.txt");
    const UNPARSEABLE: &str =
        include_str!("../../tests/fixtures/solve_errors/unparseable_network_error.txt");
    const POST_WIDEN_UNSAT: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_unsat_post_widen.txt");
    const UV_CLOSURE_TINYOBJLOADER: &str =
        include_str!("../../tests/fixtures/solve_errors/uv_closure_tinyobjloader_prerelease.txt");
    const UV_CLOSURE_CUDA_BINDINGS: &str =
        include_str!("../../tests/fixtures/solve_errors/uv_closure_cuda_bindings_widen.txt");
    // Fix #19 fixture (lock-succ-brief.md ACCEPTANCE RUN #5): a second
    // uv-closure report for the SAME cuda-bindings footer, but with a
    // sibling requirer (torchvision) alongside torch in the "Because"
    // prose -- reproduces iter-2 of the acceptance run, where torch is
    // already tried (its pypi_override was applied in iter 1) and the old
    // single-path chain walk had nowhere else to look.
    const UV_CLOSURE_CUDA_BINDINGS_ITER2_TORCHVISION: &str = include_str!(
        "../../tests/fixtures/solve_errors/uv_closure_cuda_bindings_widen_iter2_torchvision.txt"
    );
    const CONDA_INCOMPATIBLE_PYGLET: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_incompatible_pyglet.txt");
    // Real failure (step4-lock-run2.log, `pm-isaaclab` proof run 2):
    // ProtoMotions' `requirements_isaaclab.txt` (a `retread-deps-from`
    // root) pins both `wandb==0.23.0` (whose metadata requires
    // `sentry-sdk>=2.0.0`) and `sentry-sdk==1.38.0` directly -- an
    // intrinsic pypi-vs-pypi conflict with no conda pin involved at all.
    const UV_CLOSURE_DEPS_FROM_INTRINSIC_PIN: &str =
        include_str!("../../tests/fixtures/solve_errors/uv_closure_deps_from_intrinsic_pin.txt");
    // Fix #21 fixture (acceptance-final.md, verbatim
    // solve-conflicts.acceptance-final.lock-2.txt): the torchvision
    // exact-pin companion conflict that surfaced after fix #20's torch
    // override reached the closure -- direct rattler conda-solver prose,
    // pinned operator `==` (not `<`/`<=`), requiring clause `==` (not a
    // `>=`/`>` floor).
    const CONDA_INCOMPATIBLE_TORCHVISION_EXACT: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_incompatible_torchvision_exact.txt");
    // Run 17 fixture (depsfrom-proof-brief.md, verbatim
    // `.retread/solve-conflicts/`): TWO packs named in the same
    // resolvo tree, each with a single-version "would require" clause --
    // `isaaclab-2.3x-pack` under a "can be installed" (satisfiable,
    // shown for context) branch, `protomotions-deps-pack` under the
    // actual "cannot be installed" dead end that owns `pandas==3.0.3`.
    // The textually-first "would require" belongs to isaaclab-2.3x-pack;
    // the correct owner is protomotions-deps-pack.
    const CONDA_TWO_PACK_PANDAS_PYTZ: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_two_pack_pandas_pytz.txt");
    // Run 8 fixture (depsfrom-proof-brief.md, verbatim
    // `.retread/solve-conflicts/lock-2.txt`): the workspace's own
    // `setuptools >=68,<81` conda pin (declared directly in several
    // `feature.*.dependencies` tables wherever `isaaclab-2.3x-pack` is
    // composed) vs the pack's own exact companion demand
    // (`isaaclab-2.3x-pack 0.54.2 would require setuptools ==83.0.0`) --
    // a RANGE-pinned (not bare `<`/`<=`) workspace side, which the
    // existing `conda_incompatible`/`conda_incompatible_exact` regexes
    // both miss (this previously fell all the way through to "could not
    // parse solver error").
    const CONDA_INCOMPATIBLE_SETUPTOOLS_RANGE: &str =
        include_str!("../../tests/fixtures/solve_errors/conda_incompatible_setuptools_range.txt");
    const UNPARSEABLE_UV_CLOSURE_NO_WHEELS: &str =
        include_str!("../../tests/fixtures/solve_errors/unparseable_uv_closure_no_wheels.txt");
    // Real failure (step4-lock-run3.log, `pm-isaaclab` proof run 3), as it
    // would surface to `retread solve`'s parser AFTER the backend's
    // sdist-only self-heal ladder (uv_closure.rs) already exhausted both
    // the conda-route and sdist-auto-build rungs for
    // `antlr4-python3-runtime` (its own exhaustion guidance appended):
    // `hydra-core==1.3.2`'s own wildcard pin
    // (`antlr4-python3-runtime==4.9.*`) names a range with zero
    // wheel-bearing builds under `--no-build`.
    const UV_CLOSURE_NO_WHEEL_TRANSITIVE: &str =
        include_str!("../../tests/fixtures/solve_errors/uv_closure_no_wheel_transitive.txt");
    // Run 13 fixture (depsfrom-proof-brief.md, verbatim
    // `.retread/solve-conflicts/lock-3.txt`): the setuptools
    // `CondaRangeVsPackPin` repair stuck (the pack's auto-routed pin
    // FOLLOWED the injected workspace range to `==80.10.2`, uv's
    // max-in-range pick), but a DEEPER conda run-dep of the same pack
    // (`dex-retargeting -> pytorch`) imposes an additional
    // `setuptools <76` cap several tree levels below the pack's own
    // top-level "would require" clause -- neither `conda_incompatible`
    // nor `conda_incompatible_exact` match (both require "cannot be
    // installed" to immediately follow the package/version with only
    // gutter/whitespace in between; this shape always has a ", which"
    // aside in the way), hence EXIT=2 (unparseable) before this fix.
    const NESTED_CONDA_CAP_PYTORCH_SETUPTOOLS: &str =
        include_str!("../../tests/fixtures/solve_errors/nested_conda_cap_pytorch_setuptools.txt");
    // Run 15 fixture (depsfrom-proof-brief.md, verbatim
    // `.retread/solve-conflicts/lock-1.txt` from the imprint workspace): a
    // NEW conflict class none of the 12 hardening fixes' specific rungs
    // recognize -- a PyPI-vs-conda-forge metadata skew. The pack's own
    // auto-routed exact pins (`moviepy ==2.2.1`, `pillow ==11.3.0`
    // elsewhere in its render, not visible in this trace) are BOTH the
    // pack's own emissions; conda-forge's `moviepy-2.2.1` recipe still
    // caps `pillow <11.0`, contradicting the pillow version the pack's
    // own uv closure (PyPI truth) already locked. No relax tier fixes
    // this -- it's what the generic fallback engine's UN-ROUTE candidate
    // exists for. `Some(Conflict::...)` is intentionally never asserted
    // for this fixture in `RegexConflictParser::parse` tests: it stays
    // unparseable to every SPECIFIC rung by design (that's the whole
    // point of the fallback engine), and is exercised end-to-end via
    // `repair::tests::end_to_end_run15_pypi_conda_metadata_skew_unroutes_moviepy`
    // instead.
    const PYPI_CONDA_METADATA_SKEW_PILLOW_MOVIEPY: &str = include_str!(
        "../../tests/fixtures/solve_errors/pypi_conda_metadata_skew_pillow_moviepy.txt"
    );

    #[test]
    fn parses_dumb_hack_shapes_and_range_amendment() {
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(CONDA_BOUNDARY_SINGLE_LINE),
            Some(Conflict::CondaBoundary {
                package: "numpy".into(),
                version: "2.3.1".into(),
            })
        );
        assert_eq!(
            p.parse(CONDA_BOUNDARY_HELP_ONLY),
            Some(Conflict::CondaBoundary {
                package: "numpy".into(),
                version: "1.26.4".into(),
            })
        );
        assert_eq!(
            p.parse(PYPI_INTERNAL),
            Some(Conflict::PypiInternal {
                package: "torch".into(),
                version: "2.7.1".into(),
            })
        );
        assert_eq!(
            p.parse(NO_CANDIDATES),
            Some(Conflict::NoCandidates {
                package: "mujoco".into(),
                version: "3.1.6".into(),
            })
        );
        assert_eq!(
            p.parse(NO_CANDIDATES_PYPI),
            Some(Conflict::NoCandidates {
                package: "torch".into(),
                version: "2.7.1".into(),
            })
        );
        assert_eq!(
            p.parse(CONDA_BOUNDARY_RANGE),
            Some(Conflict::CondaWidenNeeded {
                package: "mujoco".into(),
                op: ">=".into(),
                floor: "3.10.3".into(),
                conda_version: "3.5.0".into(),
                requiring_chain: Vec::new(),
                pack_name: None,
            })
        );
        assert_eq!(
            p.parse(CONDA_BOUNDARY_RANGE_ALT),
            Some(Conflict::CondaWidenNeeded {
                package: "foo".into(),
                op: ">".into(),
                floor: "2.10.3".into(),
                conda_version: "2.9.0".into(),
                requiring_chain: Vec::new(),
                pack_name: None,
            })
        );
    }

    #[test]
    fn parses_uv_closure_json_rpc_conflicts() {
        let p = RegexConflictParser::new();
        // Real log sample (lock-iter-2.log): uv-closure ErrorObject,
        // tinyobjloader pre-release with no matching conda constraint ->
        // treated as an intrinsic PyPI-side conflict.
        assert_eq!(
            p.parse(UV_CLOSURE_TINYOBJLOADER),
            Some(Conflict::PypiInternal {
                package: "tinyobjloader".into(),
                version: "2.0.0rc13".into(),
            })
        );
        // Real log sample (lock-succ-18.log): uv-closure ErrorObject,
        // torch's cuda-bindings>=13.0.3,<14 requirement disjoint from the
        // cuda-major-table-injected cuda-bindings>=12,<13 conda pin ->
        // widen floor is the *other* (non-conda) range in the disjunction.
        assert_eq!(
            p.parse(UV_CLOSURE_CUDA_BINDINGS),
            Some(Conflict::CondaWidenNeeded {
                package: "cuda-bindings".into(),
                op: ">=".into(),
                floor: "13.0.3".into(),
                conda_version: ">=12".into(),
                // Attribution-fix chain: cuda-bindings has no user conda
                // pin of its own; the requiring chain (torch ->
                // isaacsim-core -> isaacsim[all]) is what repair walks to
                // find `torch`'s `pytorch-gpu` conda pin instead.
                requiring_chain: vec![
                    "torch".into(),
                    "isaacsim-core".into(),
                    "isaacsim[all]".into(),
                ],
                pack_name: Some("isaac-pack-latest".into()),
            })
        );
        // Real log sample (solve-hover-gpu.log distilled): a uv-closure
        // ErrorObject that isn't a resolver conflict at all (no wheels in
        // the exported closure) -- must fail gracefully, not crash or
        // fall through to unrelated conda-prose regexes.
        assert_eq!(p.parse(UNPARSEABLE_UV_CLOSURE_NO_WHEELS), None);
    }

    #[test]
    fn requiring_chain_collects_sibling_requirers_not_just_first_match() {
        // Fix #19: torchvision is a SIBLING requirer of the same
        // cuda-bindings footer, not on torch's single path -- a strict
        // "first match wins" walk would stop at torch and never surface
        // torchvision at all. The chain must include both, in report
        // order, so repair.rs's "skip already-tried" loop can fall
        // through to torchvision once torch is exhausted.
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(UV_CLOSURE_CUDA_BINDINGS_ITER2_TORCHVISION),
            Some(Conflict::CondaWidenNeeded {
                package: "cuda-bindings".into(),
                op: ">=".into(),
                floor: "13.0.3".into(),
                conda_version: ">=12".into(),
                requiring_chain: vec![
                    "torch".into(),
                    "torchvision".into(),
                    "isaacsim-core".into(),
                    "isaacsim[all]".into(),
                ],
                pack_name: Some("isaac-pack-latest".into()),
            })
        );
    }

    #[test]
    fn parses_direct_conda_incompatible_prose() {
        let p = RegexConflictParser::new();
        // Real log sample (lock-succ-14.log): direct (non-JSON-RPC)
        // resolvo-style "cannot be installed because there are no viable
        // options" / "would require" conda-solver prose -- distinct from
        // the older "pinned by the conda solve" help-text shape.
        assert_eq!(
            p.parse(CONDA_INCOMPATIBLE_PYGLET),
            Some(Conflict::CondaWidenNeeded {
                package: "pyglet".into(),
                op: ">=".into(),
                floor: "2".into(),
                conda_version: "<2".into(),
                requiring_chain: Vec::new(),
                pack_name: None,
            })
        );
    }

    #[test]
    fn parses_exact_pin_companion_conda_conflict_to_pack_scoped_override() {
        // Fix #21: an exact `==` conda pin (torchvision==0.25.0) clashing
        // with a pack's own exact `==` companion demand
        // (isaac-pack-latest would require torchvision==0.26.0) must parse
        // to a `CondaWidenNeeded` carrying `pack_name: Some(..)` so the
        // fix-#20 pack-override path fires (workspace pin wins, pack gets
        // the override) -- NOT fall through to `unparseable`.
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(CONDA_INCOMPATIBLE_TORCHVISION_EXACT),
            Some(Conflict::CondaWidenNeeded {
                package: "torchvision".into(),
                op: "==".into(),
                floor: "0.26.0".into(),
                conda_version: "==0.25.0".into(),
                requiring_chain: Vec::new(),
                pack_name: Some("isaac-pack-latest".into()),
            })
        );
    }

    #[test]
    fn parses_range_pinned_conda_conflict_to_pack_override() {
        // Run 8 / tenth fix: unlike fix #21's exact-vs-exact companion
        // shape, here the workspace pin is a `>=`/`<` RANGE -- but this
        // fixture's conflict tree NAMES the pack (`isaaclab-2.3x-pack
        // 0.54.2 would require setuptools ==83.0.0`), so doctrine now says
        // the workspace's hand-written range is real owner intent (must
        // NOT be widened) and the pack's exact `==83.0.0` demand is the
        // auto-routed side that gets relaxed via a pypi override written
        // into the PACK's own closure, not the workspace manifest.
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(CONDA_INCOMPATIBLE_SETUPTOOLS_RANGE),
            Some(Conflict::CondaRangeVsPackPin {
                package: "setuptools".into(),
                conda_range: ">=68,<81".into(),
                pack_demand: "83.0.0".into(),
                pack_name: "isaaclab-2.3x-pack".into(),
            })
        );
    }

    #[test]
    fn parses_range_pinned_conda_conflict_with_no_named_pack_to_workspace_widen() {
        // Fallback shape (no named pack in the conflict tree, e.g. a
        // direct conda-conda clash with no pack rendering step involved):
        // still widens the workspace's own pin, same as before this fix,
        // since there is no pack to attribute an override to.
        const NO_PACK_NAME: &str = "setuptools >=68,<81 cannot be installed because there are no viable options:\n  would require setuptools ==83.0.0, which cannot be installed";
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(NO_PACK_NAME),
            Some(Conflict::CondaWidenNeeded {
                package: "setuptools".into(),
                op: ">=".into(),
                floor: "83.0.0".into(),
                conda_version: "<81".into(),
                requiring_chain: Vec::new(),
                pack_name: None,
            })
        );
    }

    #[test]
    fn parses_deps_from_intrinsic_pypi_conflict() {
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(UV_CLOSURE_DEPS_FROM_INTRINSIC_PIN),
            Some(Conflict::DepsFromPin {
                package: "sentry-sdk".into(),
                pinned_version: "1.38.0".into(),
                op: ">=".into(),
                floor: "2.0.0".into(),
                requirer: "wandb".into(),
                pack_name: Some("protomotions-deps-pack".into()),
            })
        );
    }

    #[test]
    fn parses_nested_conda_cap_beneath_pack_auto_routed_exact_pin() {
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(NESTED_CONDA_CAP_PYTORCH_SETUPTOOLS),
            Some(Conflict::NestedCondaCap {
                package: "setuptools".into(),
                pack_name: "isaaclab-2.3x-pack".into(),
                pack_demand: "80.10.2".into(),
                cap_op: "<".into(),
                cap_version: "76".into(),
                via: "pytorch".into(),
            })
        );
    }

    #[test]
    fn parses_no_wheel_transitive_conflict_after_sdist_heal_exhausted() {
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(UV_CLOSURE_NO_WHEEL_TRANSITIVE),
            Some(Conflict::NoWheelTransitive {
                package: "antlr4-python3-runtime".into(),
                range: ">=4.9,<=4.9.3".into(),
                requirer: "hydra-core".into(),
                requirer_pin: "1.3.2".into(),
                pack_name: Some("protomotions-deps-pack".into()),
            })
        );
    }

    /// Generic fallback extractor: must find the package(s) that matter
    /// in EVERY specific-shape fixture the 12-rung hardening series
    /// produced -- the whole point is that this one permissive scan
    /// covers every prose shape a new rung would otherwise be needed for.
    #[test]
    fn extract_generic_mentions_finds_relevant_packages_in_every_fixture() {
        let p = RegexConflictParser::new();
        let cases: &[(&str, &[&str])] = &[
            (CONDA_BOUNDARY_SINGLE_LINE, &["numpy"]),
            (CONDA_BOUNDARY_HELP_ONLY, &["numpy"]),
            (PYPI_INTERNAL, &["torch"]),
            (NO_CANDIDATES, &["mujoco"]),
            (NO_CANDIDATES_PYPI, &["torch"]),
            (CONDA_BOUNDARY_RANGE, &["mujoco"]),
            (CONDA_BOUNDARY_RANGE_ALT, &["foo"]),
            (UV_CLOSURE_TINYOBJLOADER, &["tinyobjloader"]),
            (UV_CLOSURE_CUDA_BINDINGS, &["cuda-bindings", "torch"]),
            (CONDA_INCOMPATIBLE_PYGLET, &["pyglet"]),
            (CONDA_INCOMPATIBLE_TORCHVISION_EXACT, &["torchvision"]),
            (CONDA_INCOMPATIBLE_SETUPTOOLS_RANGE, &["setuptools"]),
            (UV_CLOSURE_DEPS_FROM_INTRINSIC_PIN, &["sentry-sdk", "wandb"]),
            (
                UV_CLOSURE_NO_WHEEL_TRANSITIVE,
                &["antlr4-python3-runtime", "hydra-core"],
            ),
            // Note: `pytorch` itself is never given a `<name> <op><ver>`
            // clause in this fixture (it only ever appears as a bare
            // "pytorch 2.7.0 | 2.7.0 | ... would require" subject, or
            // wildcard "pytorch *" -- neither carries an operator), so
            // it isn't expected as a mention here; `setuptools` (the
            // package actually in conflict) is.
            (
                NESTED_CONDA_CAP_PYTORCH_SETUPTOOLS,
                &["setuptools", "dex-retargeting"],
            ),
            (
                PYPI_CONDA_METADATA_SKEW_PILLOW_MOVIEPY,
                &["moviepy", "pillow"],
            ),
        ];
        for (fixture, expected) in cases {
            let mentions = p.extract_generic_mentions(fixture);
            let found: Vec<String> = mentions
                .iter()
                .map(|m| m.package.to_ascii_lowercase())
                .collect();
            for want in *expected {
                assert!(
                    found.iter().any(|f| f == &want.to_ascii_lowercase()),
                    "expected to find {want} in {found:?} (fixture head: {:?})",
                    tail(fixture, 200)
                );
            }
        }
    }

    #[test]
    fn pypi_conda_metadata_skew_stays_unparseable_to_every_specific_rung() {
        // The whole point of the generic fallback engine: this shape must
        // NOT gain a 13th specific rung. `repair.rs`'s end-to-end test
        // exercises the fallback engine directly against this fixture.
        let p = RegexConflictParser::new();
        assert_eq!(p.parse(PYPI_CONDA_METADATA_SKEW_PILLOW_MOVIEPY), None);
    }

    #[test]
    fn extract_bundle_name_reads_the_uv_closure_bundle_label() {
        let p = RegexConflictParser::new();
        assert_eq!(
            p.extract_bundle_name(UV_CLOSURE_DEPS_FROM_INTRINSIC_PIN),
            Some("protomotions-deps-pack".to_string())
        );
        // No JSON-RPC bundle label at all in this fixture, but the
        // outermost "<name> <ver> would require" clause names a real
        // pack (`isaaclab-hover-pack`) -- the fallback finds it too.
        assert_eq!(
            p.extract_bundle_name(CONDA_INCOMPATIBLE_PYGLET),
            Some("isaaclab-hover-pack".to_string())
        );
        assert_eq!(
            p.extract_bundle_name(NESTED_CONDA_CAP_PYTORCH_SETUPTOOLS),
            Some("isaaclab-2.3x-pack".to_string())
        );
    }

    /// Run-17 wrong-owner bug: when two packs are each named with their
    /// own "would require" clause, the actual dead-end pack (marked
    /// "cannot be installed") must win over a merely-satisfiable pack
    /// shown for context (marked "can be installed"), even though the
    /// satisfiable pack's clause appears FIRST in the text.
    #[test]
    fn extract_bundle_name_prefers_the_dead_end_pack_over_a_satisfiable_one_shown_first() {
        let p = RegexConflictParser::new();
        assert_eq!(
            p.extract_bundle_name(CONDA_TWO_PACK_PANDAS_PYTZ),
            Some("protomotions-deps-pack".to_string())
        );
    }

    #[test]
    fn strips_ansi_before_parsing_and_rejects_unparseable() {
        let p = RegexConflictParser::new();
        let colored = "\u{1b}[31mNo candidates were found for numpy == 2.3.1\u{1b}[0m";
        assert_eq!(
            p.parse(&p.strip_ansi(colored)),
            Some(Conflict::NoCandidates {
                package: "numpy".into(),
                version: "2.3.1".into(),
            })
        );
        assert_eq!(p.parse(UNPARSEABLE), None);
        assert!(RegexConflictParser::is_post_widen_conda_unsat(
            POST_WIDEN_UNSAT
        ));
    }
}
