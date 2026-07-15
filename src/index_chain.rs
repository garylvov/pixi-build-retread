/// Purpose for which a PyPI index chain is assembled.
pub(crate) enum IndexPurpose {
    /// Root resolution preserves entry-specific index priority before the
    /// complete workspace chain.
    RootResolve,
    /// Transitive fallback consults the complete workspace chain before an
    /// entry-specific index that may only host the root package.
    TransitiveFallback,
}

/// Canonical public PyPI Simple index, used only as a safe default when the
/// caller could not obtain a complete workspace chain.
pub(crate) const PUBLIC_PYPI: &str = crate::workspace::DEFAULT_PYPI_INDEX;

/// Build an ordered PyPI index universe for one resolution boundary.
///
/// `workspace` is expected to be the complete chain produced by
/// `WorkspaceManifest::resolution_pypi_index_urls`, including either pixi's
/// implicit public default or its explicit replacement. Consequently this
/// helper must not append public PyPI after an explicit `index-url` override.
/// URLs are deduplicated without regard to trailing slashes.
pub(crate) fn index_chain(
    entry_indexes: impl IntoIterator<Item = String>,
    workspace: &[String],
    purpose: IndexPurpose,
) -> Vec<String> {
    fn push_unique(indexes: &mut Vec<String>, index: String) {
        if !indexes
            .iter()
            .any(|existing| existing.trim_end_matches('/') == index.trim_end_matches('/'))
        {
            indexes.push(index);
        }
    }

    let entry_indexes = entry_indexes.into_iter().collect::<Vec<_>>();
    let workspace_indexes = if workspace.is_empty() {
        vec![PUBLIC_PYPI.to_string()]
    } else {
        workspace.to_vec()
    };

    let mut indexes = Vec::new();
    let mut append = |source: &[String]| {
        for index in source {
            push_unique(&mut indexes, index.clone());
        }
    };
    match purpose {
        IndexPurpose::RootResolve => {
            append(&entry_indexes);
            append(&workspace_indexes);
        }
        IndexPurpose::TransitiveFallback => {
            append(&workspace_indexes);
            append(&entry_indexes);
        }
    }
    indexes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_resolution_keeps_entry_priority() {
        let indexes = index_chain(
            ["https://entry.example/simple".to_string()],
            &[
                "https://pypi.org/simple".to_string(),
                "https://extra.example/simple".to_string(),
            ],
            IndexPurpose::RootResolve,
        );
        assert_eq!(
            indexes,
            vec![
                "https://entry.example/simple".to_string(),
                "https://pypi.org/simple".to_string(),
                "https://extra.example/simple".to_string(),
            ]
        );
    }

    #[test]
    fn transitive_fallback_keeps_workspace_priority() {
        let indexes = index_chain(
            ["https://entry.example/simple".to_string()],
            &[
                "https://pypi.org/simple".to_string(),
                "https://extra.example/simple".to_string(),
            ],
            IndexPurpose::TransitiveFallback,
        );
        assert_eq!(
            indexes,
            vec![
                "https://pypi.org/simple".to_string(),
                "https://extra.example/simple".to_string(),
                "https://entry.example/simple".to_string(),
            ]
        );
    }

    #[test]
    fn explicit_workspace_chain_is_not_extended_with_public_pypi() {
        let indexes = index_chain(
            ["https://entry.example/simple/".to_string()],
            &["https://packages.example/simple".to_string()],
            IndexPurpose::TransitiveFallback,
        );
        assert_eq!(
            indexes,
            vec![
                "https://packages.example/simple".to_string(),
                "https://entry.example/simple/".to_string(),
            ]
        );
        assert!(!indexes.iter().any(|index| index == PUBLIC_PYPI));
    }

    #[test]
    fn empty_workspace_chain_safely_uses_public_pypi() {
        assert_eq!(
            index_chain(std::iter::empty(), &[], IndexPurpose::TransitiveFallback),
            vec![PUBLIC_PYPI.to_string()]
        );
    }
}
