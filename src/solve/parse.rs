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
    },
}

impl Conflict {
    pub fn kind(&self) -> &'static str {
        match self {
            Conflict::NoCandidates { .. } => "NoCandidates",
            Conflict::CondaBoundary { .. } => "CondaBoundary",
            Conflict::PypiInternal { .. } => "PypiInternal",
            Conflict::CondaWidenNeeded { .. } => "CondaWidenNeeded",
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
        }
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
            })
        );
        assert_eq!(
            p.parse(CONDA_BOUNDARY_RANGE_ALT),
            Some(Conflict::CondaWidenNeeded {
                package: "foo".into(),
                op: ">".into(),
                floor: "2.10.3".into(),
                conda_version: "2.9.0".into(),
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
