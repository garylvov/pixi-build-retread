//! p6z. The lenient `Requires-Dist` reader retread was missing, and the 26
//! roots it cost.
//!
//! THE DEFECT THIS EXISTS FOR (boarded p6w-1). `holosoma-pack`'s 26 detected
//! roots were dropped by the resolve-time back-off in jobs 5764452/5764453 and
//! again in 5776669/5776671 -- and uv never ran. The failure was retread's own
//! reader:
//!
//! ```text
//! computing uv closure for bundle `holosoma-pack`: parsing requirement
//! `PyYAML (>=5.1.*)`: Operator >= cannot be used with a wildcard version
//! specifier
//! ```
//!
//! `PyYAML (>=5.1.*)` is `omegaconf==2.0.6`'s real `Requires-Dist` (verified
//! against the PyPI JSON API for that exact version). It is not valid PEP 440:
//! a wildcard may follow `==` and `!=` only. `uv_pep508::Requirement::from_str`
//! is right to refuse it, and every one of retread's metadata readers called
//! exactly that.
//!
//! MEASURED, because the alternative was to call the distribution defective.
//! `uv 0.11.29` on this box, on a project whose only dependency is
//! `omegaconf==2.0.6`:
//!
//! ```text
//! Resolved 4 packages   ->  pyyaml 6.0.3
//! ```
//!
//! and, pinned against it, uv names its own normalization in its own prose:
//!
//! ```text
//! Because omegaconf==2.0.6 depends on pyyaml>=5.1 and your project depends
//! on omegaconf==2.0.6, we can conclude that your project depends on
//! pyyaml>=5.1.
//! ```
//!
//! So uv reads `>=5.1.*` as `>=5.1`. pip does the same. The distribution is
//! not the defect; retread's reader was. This module is that reader.
//!
//! WHY THE RULES ARE COPIED RATHER THAN IMPORTED. uv keeps them in
//! `uv-pypi-types` (`LenientRequirement`, `crates/uv-pypi-types/src/
//! lenient_requirement.rs` at tag `0.9.5`, the tag this crate already pins for
//! `uv-pep508`). Taking that crate as a dependency would pull `rkyv`, `jiff`,
//! `petgraph`, `mailparse` and `schemars`, and would force the `rkyv` feature
//! on the `uv-pep508` we already build -- a large build-graph change for eight
//! regular expressions. The regexes, their order, and their messages below are
//! byte-for-byte uv's, and uv's own test corpus for them is reproduced in this
//! module's tests so a divergence fails here rather than in a lock.
//!
//! WHAT A CALLER GETS. A repair is announced (`tracing::warn!`), never silent,
//! and a requirement that cannot be repaired returns [`LenientParseError`],
//! which renders the offending clause. Callers that know which distribution
//! published the line add it with `anyhow::Context`, so the row names the
//! distribution AND the clause instead of a bare parse error with no owner --
//! the second half of p6w-1: the holosoma text never said whose
//! `Requires-Dist` failed, so 25 bystanders went with the one culprit.

use std::borrow::Cow;
use std::str::FromStr;
use std::sync::LazyLock;

use regex::Regex;

/// Ex) `>=7.2.0<8.0.0`
static MISSING_COMMA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d)([<>=~^!])").unwrap());
/// Ex) `!=~5.0`
static NOT_EQUAL_TILDE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!=~((?:\d\.)*\d)").unwrap());
/// Ex) `>=1.9.*`, `<3.4.*` -- the omegaconf 2.0.6 shape.
static INVALID_TRAILING_DOT_STAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(<=|>=|<|>)(\d+(\.\d+)*)\.\*").unwrap());
/// Ex) `!=3.0*`
static MISSING_DOT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d\.\d)+\*").unwrap());
/// Ex) `>=3.6,`
static TRAILING_COMMA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r",\s*$").unwrap());
/// Ex) `>dev`
static GREATER_THAN_DEV: LazyLock<Regex> = LazyLock::new(|| Regex::new(r">dev").unwrap());
/// Ex) `>=9.0.0a1.0`
static TRAILING_ZERO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+(\.\d)*(a|b|rc|post|dev)\d+)\.0").unwrap());

type FixUp = for<'a> fn(&'a str) -> Cow<'a, str>;

/// The repair ladder, in uv's order. Order is load-bearing: each fixup is
/// applied to the output of the previous one, so a line with two defects is
/// repaired cumulatively.
static FIXUPS: &[(FixUp, &str)] = &[
    (
        |input| MISSING_COMMA.replace_all(input, r"$1,$2"),
        "inserting missing comma",
    ),
    (
        |input| NOT_EQUAL_TILDE.replace_all(input, r"!=${1}.*"),
        "replacing invalid tilde with wildcard",
    ),
    (
        |input| INVALID_TRAILING_DOT_STAR.replace_all(input, r"${1}${2}"),
        "removing star after comparison operator other than equal and not equal",
    ),
    (
        |input| MISSING_DOT.replace_all(input, r"${1}.*"),
        "inserting missing dot",
    ),
    (
        |input| TRAILING_COMMA.replace_all(input, r"${1}"),
        "removing trailing comma",
    ),
    (
        |input| GREATER_THAN_DEV.replace_all(input, r">0.0.0dev"),
        "assuming 0.0.0dev",
    ),
    (
        |input| TRAILING_ZERO.replace_all(input, r"${1}"),
        "removing trailing zero",
    ),
    (remove_stray_quotes, "removing stray quotes"),
];

/// Given `>= 2.7'`, rewrite to `>= 2.7`, without touching a marker's quotes.
fn remove_stray_quotes(input: &str) -> Cow<'_, str> {
    static STRAY_QUOTES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"['"]"#).unwrap());
    match input.find(';') {
        Some(markers) => {
            let requirement = STRAY_QUOTES.replace_all(&input[..markers], "");
            format!("{}{}", requirement, &input[markers..]).into()
        }
        None => STRAY_QUOTES.replace_all(input, ""),
    }
}

/// A `Requires-Dist` line no repair in the ladder could make parseable.
///
/// The rendering carries the CLAUSE verbatim. The DISTRIBUTION that published
/// it is added by the caller that knows it -- see this module's header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LenientParseError {
    pub raw: String,
    pub message: String,
}

impl std::fmt::Display for LenientParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parsing requirement `{}`: {}",
            self.raw.trim(),
            self.message
        )
    }
}

impl std::error::Error for LenientParseError {}

/// What a successful lenient parse had to do to the line, if anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Repair {
    /// The line as the distribution published it.
    pub before: String,
    /// The line that actually parsed.
    pub after: String,
    /// uv's own names for the repairs applied, in order.
    pub fixups: Vec<&'static str>,
}

/// Parse one `Requires-Dist` value, repairing the legacy/invalid shapes uv and
/// pip repair, and reporting whether a repair was needed.
///
/// Strict first: a valid line is never rewritten, so this cannot change the
/// meaning of metadata that was already well formed.
pub fn parse_requirement_lenient_detailed(
    raw: &str,
) -> Result<(uv_pep508::Requirement, Option<Repair>), LenientParseError> {
    match uv_pep508::Requirement::from_str(raw) {
        Ok(requirement) => Ok((requirement, None)),
        Err(first_error) => {
            let mut patched = raw.to_string();
            let mut applied: Vec<&'static str> = Vec::new();
            for (fixup, message) in FIXUPS {
                let next = fixup(patched.as_ref());
                if next == patched {
                    continue;
                }
                applied.push(*message);
                let next = next.to_string();
                if let Ok(requirement) = uv_pep508::Requirement::from_str(&next) {
                    return Ok((
                        requirement,
                        Some(Repair {
                            before: raw.to_string(),
                            after: next,
                            fixups: applied,
                        }),
                    ));
                }
                patched = next;
            }
            Err(LenientParseError {
                raw: raw.to_string(),
                message: first_error.to_string(),
            })
        }
    }
}

/// [`parse_requirement_lenient_detailed`], announcing any repair on the
/// backend's own timeline.
///
/// A repair is a fact about upstream metadata an operator may need to see, so
/// it is a WARN with both spellings of the line, not a silent rewrite.
pub fn parse_requirement_lenient(
    raw: &str,
) -> Result<uv_pep508::Requirement, LenientParseError> {
    let (requirement, repair) = parse_requirement_lenient_detailed(raw)?;
    if let Some(repair) = repair {
        tracing::warn!(
            before = %repair.before,
            after = %repair.after,
            fixups = %repair.fixups.join(", "),
            "requires_dist_repaired: an invalid PEP 508 requirement was normalized the way uv and pip normalize it",
        );
    }
    Ok(requirement)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(raw: &str) -> uv_pep508::Requirement {
        parse_requirement_lenient(raw).expect("lenient parse must succeed")
    }

    /// THE MEASURED ROW. `omegaconf==2.0.6`'s real `Requires-Dist`, the exact
    /// string in jobs 5764452/5776669's holosoma failure, and the version uv
    /// 0.11.29 resolves it to on this box.
    #[test]
    fn p6z_a_omegaconf_2_0_6_legacy_wildcard_parses_as_uv_reads_it() {
        let strict: Result<uv_pep508::Requirement, _> =
            uv_pep508::Requirement::from_str("PyYAML (>=5.1.*)");
        assert!(
            strict.is_err(),
            "non-vacuity: the strict reader must still refuse this line, or the \
             lenient path proves nothing",
        );

        let (requirement, repair) =
            parse_requirement_lenient_detailed("PyYAML (>=5.1.*)").expect("uv accepts this line");
        assert_eq!(requirement.name.to_string(), "pyyaml");
        let repair = repair.expect("the line had to be repaired");
        assert_eq!(repair.after, "PyYAML (>=5.1)");
        assert!(
            repair.fixups.contains(
                &"removing star after comparison operator other than equal and not equal"
            ),
            "{:?}",
            repair.fixups
        );

        // uv's measured semantics: `>=5.1`, NOT `>=5.1,<5.2`. uv 0.11.29 picks
        // pyyaml 6.0.3 for this requirement and refuses a `pyyaml==5.0.1` pin
        // with `omegaconf==2.0.6 depends on pyyaml>=5.1`.
        let Some(uv_pep508::VersionOrUrl::VersionSpecifier(specifiers)) =
            requirement.version_or_url.as_ref()
        else {
            panic!("expected version specifiers");
        };
        assert_eq!(specifiers.to_string(), ">=5.1");
        let six = uv_pep508::uv_pep440::Version::from_str("6.0.3").unwrap();
        let five_oh_one = uv_pep508::uv_pep440::Version::from_str("5.0.1").unwrap();
        assert!(specifiers.contains(&six), "uv resolved this to pyyaml 6.0.3");
        assert!(
            !specifiers.contains(&five_oh_one),
            "uv refused pyyaml==5.0.1 against this very requirement",
        );
    }

    /// A VALID line is never touched. Without this the repair ladder could
    /// silently rewrite well-formed metadata and nothing would notice.
    #[test]
    fn p6z_a_valid_requirement_is_never_rewritten() {
        for line in [
            "numpy>=1.26.4,<2",
            "torch==2.9.1 ; python_version == '3.11'",
            "setuptools",
            "jax[cuda12]==0.7.1",
            "pyyaml==5.1.*",
            "pyyaml!=5.1.*",
        ] {
            let (_, repair) = parse_requirement_lenient_detailed(line)
                .unwrap_or_else(|e| panic!("`{line}` must parse strictly: {e}"));
            assert_eq!(repair, None, "`{line}` must not be repaired");
        }
    }

    /// uv's own corpus for `LenientRequirement`
    /// (`crates/uv-pypi-types/src/lenient_requirement.rs`, tag 0.9.5). A
    /// divergence from uv must fail HERE, not in a lock.
    #[test]
    fn p6z_a_uvs_own_lenient_corpus_round_trips() {
        let cases: &[(&str, &str)] = &[
            ("elasticsearch-dsl (>=7.2.0<8.0.0)", "elasticsearch-dsl>=7.2.0,<8.0.0"),
            ("jupyter-core (!=~5.0,>=4.12)", "jupyter-core!=5.0.*,>=4.12"),
            ("jupyter-core (!=~5,>=4.12)", "jupyter-core!=5.*,>=4.12"),
            ("torch (>=1.9.*)", "torch>=1.9"),
            ("pyzmq (>=2.7,!=3.0*,!=3.1*,!=3.2*)", "pyzmq>=2.7,!=3.0.*,!=3.1.*,!=3.2.*"),
            ("pyzmq >=3.6,", "pyzmq>=3.6"),
            ("botocore>=1.3.0,<1.4.0',", "botocore>=1.3.0,<1.4.0"),
        ];
        for (input, expected_name_and_specs) in cases {
            let requirement = parsed(input);
            let rendered = match requirement.version_or_url.as_ref() {
                Some(uv_pep508::VersionOrUrl::VersionSpecifier(specs)) => {
                    format!("{}{}", requirement.name, specs)
                }
                _ => requirement.name.to_string(),
            };
            // uv's `VersionSpecifiers` Display puts a space after each comma
            // and orders clauses canonically; the corpus is written the way a
            // wheel spells it. Compare the MEANING -- the name plus the SET of
            // clauses -- not the rendering.
            let clauses = |rendered: &str| {
                let (name, specs) = rendered
                    .split_once(|c: char| c == '>' || c == '<' || c == '!' || c == '=' || c == '~')
                    .map_or((rendered.to_string(), String::new()), |(name, _)| {
                        (
                            name.to_string(),
                            rendered[name.len()..].replace(' ', ""),
                        )
                    });
                let mut set: Vec<String> =
                    specs.split(',').map(str::to_string).filter(|s| !s.is_empty()).collect();
                set.sort();
                (name, set)
            };
            assert_eq!(
                clauses(&rendered),
                clauses(expected_name_and_specs),
                "uv reads `{input}` as `{expected_name_and_specs}`, got `{rendered}`",
            );
        }

        // Markers survive the quote stripper.
        let with_marker = parsed("numpy \">=1.19\"; python_version >= \"3.7\"");
        assert_eq!(with_marker.name.to_string(), "numpy");
        assert!(
            !with_marker.marker.is_true(),
            "the marker must survive quote removal",
        );
    }

    /// A line no fixup can rescue still FAILS, and the failure carries the
    /// clause. A lenient reader that swallowed everything would be the same
    /// defect in the other direction.
    #[test]
    fn p6z_a_an_unrepairable_line_still_fails_and_names_the_clause() {
        let error = parse_requirement_lenient("=== not a requirement ===")
            .expect_err("garbage must not parse");
        assert!(
            error.to_string().contains("=== not a requirement ==="),
            "{error}",
        );
    }
}
