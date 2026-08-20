//! Co-resident sibling bundle pins, read from an installed prefix.
//!
//! Two Retread bundles activated into ONE Pixi environment each solve their
//! own closure. `sibling_lock_constraints` (handler/mod.rs) makes the second
//! solve honour the first one's pins -- but only once that sibling lock
//! EXISTS. On a cold first pass neither lock is written yet, so both packs
//! pin their shared names independently and can disagree (F29); a second
//! relock, run when both locks are on disk, converges.
//!
//! Until it does, whichever bundle replays last wins the prefix and the
//! loser's verify reports its wheels as "missing". That name is wrong: the
//! dist is present, at the CO-RESIDENT sibling's pin. This module is the
//! single reader of the co-resident locks so both halves -- the build-time
//! warning and the install-time verify text -- say the same true thing.

use std::collections::BTreeMap;
use std::path::Path;

use crate::installer::normalize_dist_name;

/// The cold-pass build warning: a co-activated sibling has no lock yet, so
/// this solve cannot see its pins. One line per sibling bundle.
pub fn sibling_lock_absent_warning(sibling: &str) -> String {
    format!(
        "co-resident sibling {sibling} has no lock yet; pins may diverge on \
         this cold pass — a second lock converges"
    )
}

/// One co-resident bundle's pin for a shared dist name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoResidentPin {
    pub bundle: String,
    pub version: String,
}

/// Every dist pinned by a bundle OTHER than `self_bundle` whose lock sidecar
/// is installed under `<prefix>/share/retread`.
///
/// Parsed as raw JSON, not `RetreadLock`: this is a diagnostic reader that
/// must still work when a sibling lock is from another schema or fails
/// replay validation. A lock it cannot read contributes nothing.
pub fn coresident_pins(prefix: &Path, self_bundle: &str) -> BTreeMap<String, Vec<CoResidentPin>> {
    let mut out: BTreeMap<String, Vec<CoResidentPin>> = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(prefix.join("share").join("retread")) else {
        return out;
    };
    let mut locks: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".lock.json"))
        })
        .collect();
    locks.sort();
    for path in locks {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        let Some(bundle) = json.get("bundle").and_then(|v| v.as_str()) else {
            continue;
        };
        if bundle == self_bundle {
            continue;
        }
        let Some(wheels) = json.get("wheels").and_then(|v| v.as_array()) else {
            continue;
        };
        for wheel in wheels {
            let (Some(name), Some(version)) = (
                wheel.get("name").and_then(|v| v.as_str()),
                wheel.get("version").and_then(|v| v.as_str()),
            ) else {
                continue;
            };
            let pins = out.entry(normalize_dist_name(name)).or_default();
            let pin = CoResidentPin {
                bundle: bundle.to_string(),
                version: version.to_string(),
            };
            if !pins.contains(&pin) {
                pins.push(pin);
            }
        }
    }
    out
}

/// Split a verify "missing" list into the entries a co-resident sibling
/// explains and the entries that are genuinely absent.
///
/// A missing `name==a` is explained when the prefix holds that name at some
/// version `b`, and a co-resident bundle's lock pins the same name at exactly
/// `b`. Then nothing is missing at all: two co-scoped packs simply pinned the
/// same shared dist differently and the sibling's pin is the one on disk.
/// This is still a FAILURE -- the bundle's own payload is not installed -- it
/// is just not the failure "missing locked wheel(s)" names.
pub fn explain_divergent_pins(
    missing: &[String],
    installed_versions: &BTreeMap<String, Vec<String>>,
    pins: &BTreeMap<String, Vec<CoResidentPin>>,
) -> (Vec<String>, Vec<String>) {
    let mut divergent = Vec::new();
    let mut unexplained = Vec::new();
    for item in missing {
        let Some((name, locked)) = item.split_once("==") else {
            unexplained.push(item.clone());
            continue;
        };
        let key = normalize_dist_name(name);
        let present = installed_versions.get(&key);
        let explained = present.and_then(|versions| {
            versions.iter().find_map(|installed| {
                pins.get(&key)?
                    .iter()
                    .find(|pin| &pin.version == installed)
                    .map(|pin| (pin.bundle.clone(), installed.clone()))
            })
        });
        match explained {
            Some((sibling, installed)) => divergent.push(format!(
                "shared dist {name}: this bundle pins {locked}, co-resident {sibling} pins \
                 {installed} and installed {installed}; divergent co-resident pins — re-lock \
                 to converge (cold-pass race)"
            )),
            None => unexplained.push(item.clone()),
        }
    }
    (divergent, unexplained)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pins() -> BTreeMap<String, Vec<CoResidentPin>> {
        BTreeMap::from([(
            "click".to_string(),
            vec![CoResidentPin {
                bundle: "protomotions-deps-pack".to_string(),
                version: "8.4.2".to_string(),
            }],
        )])
    }

    #[test]
    fn a_sibling_pin_that_matches_the_installed_version_renames_the_failure() {
        let installed = BTreeMap::from([("click".to_string(), vec!["8.4.2".to_string()])]);
        let (divergent, unexplained) =
            explain_divergent_pins(&["click==8.1.7".to_string()], &installed, &pins());
        assert!(unexplained.is_empty(), "unexplained was {unexplained:?}");
        assert_eq!(divergent.len(), 1);
        let message = &divergent[0];
        for needle in [
            "click",
            "this bundle pins 8.1.7",
            "co-resident protomotions-deps-pack pins 8.4.2",
            "installed 8.4.2",
            "re-lock to converge (cold-pass race)",
        ] {
            assert!(
                message.contains(needle),
                "{needle:?} missing from {message}"
            );
        }
    }

    #[test]
    fn an_absent_dist_stays_missing_and_a_third_party_version_is_not_blamed_on_a_sibling() {
        let installed = BTreeMap::from([("click".to_string(), vec!["7.0".to_string()])]);
        let (divergent, unexplained) = explain_divergent_pins(
            &["click==8.1.7".to_string(), "scipy==1.15.3".to_string()],
            &installed,
            &pins(),
        );
        assert!(divergent.is_empty(), "divergent was {divergent:?}");
        assert_eq!(unexplained, vec!["click==8.1.7", "scipy==1.15.3"]);
    }

    #[test]
    fn coresident_pins_read_sibling_locks_and_skip_the_bundles_own() {
        let dir = std::env::temp_dir().join(format!(
            "retread-coresident-pins-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let share = dir.join("share").join("retread");
        std::fs::create_dir_all(&share).unwrap();
        std::fs::write(
            share.join("retread-protomotions-deps-pack.lock.json"),
            r#"{"bundle":"protomotions-deps-pack","wheels":[{"name":"Click","version":"8.4.2"}]}"#,
        )
        .unwrap();
        std::fs::write(
            share.join("retread-isaaclab-2.3x-pack.lock.json"),
            r#"{"bundle":"isaaclab-2.3x-pack","wheels":[{"name":"click","version":"8.1.7"}]}"#,
        )
        .unwrap();
        std::fs::write(share.join("not-a-lock.json"), "{}").unwrap();

        let found = coresident_pins(&dir, "isaaclab-2.3x-pack");
        assert_eq!(
            found.get("click"),
            Some(&vec![CoResidentPin {
                bundle: "protomotions-deps-pack".to_string(),
                version: "8.4.2".to_string(),
            }]),
            "found was {found:?}",
        );
        assert_eq!(found.len(), 1, "the bundle's own lock must not be a pin");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_cold_pass_warning_names_the_sibling_and_the_remedy() {
        let warning = sibling_lock_absent_warning("protomotions-deps-pack");
        assert!(warning.contains("co-resident sibling protomotions-deps-pack has no lock yet"));
        assert!(warning.contains("pins may diverge on this cold pass"));
        assert!(warning.contains("a second lock converges"), "{warning}");
    }
}
