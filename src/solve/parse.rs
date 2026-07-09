use regex::Regex;

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
            conda_incompatible_range: Regex::new(
                r"(?s)([a-zA-Z0-9_-]+)\s*(?:>=|>)\s*[0-9][0-9a-zA-Z.]*\s*,\s*(<|<=)\s*([0-9][0-9a-zA-Z.]*)[\s│├╰└─▶]+cannot\s+be\s+installed[\s│├╰└─▶]+because\s+there\s+are\s+no\s+viable\s+options",
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
            let conda_op = caps[2].to_string();
            let conda_version = caps[3].to_string();
            let escaped = regex::escape(&package);
            let would_require = Regex::new(&format!(
                r"(?s)would require[\s│├╰└─▶]*{escaped}\s*(==|=|>=|>)\s*([0-9][0-9a-zA-Z.]*)"
            ))
            .ok()?;
            if let Some(rcaps) = would_require.captures(stderr) {
                return Some(Conflict::CondaWidenNeeded {
                    package,
                    op: ">=".to_string(),
                    floor: rcaps[2].to_string(),
                    conda_version: format!("{conda_op}{conda_version}"),
                    requiring_chain: Vec::new(),
                    pack_name: None,
                });
            }
        }

        None
    }

    pub fn strip_ansi<'a>(&self, stderr: &'a str) -> std::borrow::Cow<'a, str> {
        self.ansi.replace_all(stderr, "")
    }

    pub fn is_post_widen_conda_unsat(stderr: &str) -> bool {
        stderr.contains("Cannot solve the request because of")
            || stderr.contains("The following packages are incompatible")
    }
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
    // `antlr4-python3-runtime` (its own exhaustion guidance appended,
    // `sdist_only_no_route_message`): `hydra-core==1.3.2`'s own wildcard
    // pin (`antlr4-python3-runtime==4.9.*`) names a range with zero
    // wheel-bearing builds under `--no-build`.
    const UV_CLOSURE_NO_WHEEL_TRANSITIVE: &str =
        include_str!("../../tests/fixtures/solve_errors/uv_closure_no_wheel_transitive.txt");

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
    fn parses_range_pinned_conda_conflict_to_workspace_widen() {
        // Run 8: unlike fix #21's exact-vs-exact companion shape (which
        // routes to a PACK override because the workspace's own pin is
        // exact and therefore authoritative), here the workspace pin is a
        // `>=`/`<` RANGE -- a manual cap chosen for an unrelated reason,
        // not a hard anchor -- so conda-as-truth widens the WORKSPACE's
        // own pin instead (`pack_name: None`, same routing as the plain
        // `conda_incompatible` shape). The pack's exact `==83.0.0` demand
        // is normalized to a `>=` floor, not re-emitted as a new exact pin.
        let p = RegexConflictParser::new();
        assert_eq!(
            p.parse(CONDA_INCOMPATIBLE_SETUPTOOLS_RANGE),
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
