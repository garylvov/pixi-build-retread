/// Purpose for which a PyPI index chain is assembled.
pub(crate) enum IndexPurpose {
    /// Resolution and restore fetches must share the complete index universe.
    Resolve,
}

/// Canonical public PyPI Simple index, retained as the terminal fallback.
pub(crate) const PUBLIC_PYPI: &str = crate::workspace::DEFAULT_PYPI_INDEX;

/// Build the ordered PyPI index universe for resolution and restore fetches.
///
/// Explicit entry indexes lead, followed by workspace indexes in their declared
/// order. URLs are deduplicated without regard to trailing slashes, and public
/// PyPI is appended unless an equivalent URL is already present.
pub(crate) fn index_chain(
    entry_indexes: impl IntoIterator<Item = String>,
    workspace: &[String],
    _purpose: IndexPurpose,
) -> Vec<String> {
    fn push_unique(indexes: &mut Vec<String>, index: String) {
        if !indexes
            .iter()
            .any(|existing| existing.trim_end_matches('/') == index.trim_end_matches('/'))
        {
            indexes.push(index);
        }
    }

    let mut indexes = Vec::new();
    for index in entry_indexes {
        push_unique(&mut indexes, index);
    }
    for index in workspace {
        push_unique(&mut indexes, index.clone());
    }
    push_unique(&mut indexes, PUBLIC_PYPI.to_string());
    indexes
}
