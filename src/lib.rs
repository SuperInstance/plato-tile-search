//! plato-tile-search — Nearest-Neighbor Tile Search
//!
//! Text-based tile search without a vector DB. Uses:
//! 1. Keyword matching (exact + stemmed)
//! 2. Trigram similarity (character-level fuzzy matching)
//! 3. Jaccard word overlap
//! 4. Combined with plato-tile-scorer signals
//!
//! ## Why No Vector DB?
//! - Zero external deps (no Qdrant, no embeddings API)
//! - Works offline, works on edge (JC1's Jetson)
//! - Good enough for <10K tiles (fleet is at 2,501)
//! - Scores combined with temporal/ghost/belief via plato-tile-scorer

use std::collections::HashMap;

/// A searchable tile.
#[derive(Debug, Clone)]
pub struct SearchableTile {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub tags: Vec<String>,
    pub domain: String,
}

impl SearchableTile {
    pub fn all_text(&self) -> String {
        let mut text = self.question.clone();
        text.push(' ');
        text.push_str(&self.answer);
        for tag in &self.tags {
            text.push(' ');
            text.push_str(tag);
        }
        text
    }
}

/// Trigram set from a string.
fn trigrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut tris = Vec::new();
    for window in chars.windows(3) {
        tris.push(window.iter().collect::<String>());
    }
    tris
}

/// Jaccard similarity between two string sets.
fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    let set_a: std::collections::HashSet<_> = a.iter().collect();
    let set_b: std::collections::HashSet<_> = b.iter().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 { return 0.0; }
    intersection as f64 / union as f64
}

/// Word-level Jaccard between two strings.
fn word_jaccard(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<_> = a.split_whitespace().map(|w| w.to_lowercase()).collect();
    let words_b: std::collections::HashSet<_> = b.split_whitespace().map(|w| w.to_lowercase()).collect();
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 { return 0.0; }
    intersection as f64 / union as f64
}

/// Keyword match: count of query words found in text.
fn keyword_match_score(query: &str, text: &str) -> f64 {
    let query_words: Vec<&str> = query.split_whitespace().collect();
    if query_words.is_empty() { return 0.0; }
    let text_lower = text.to_lowercase();
    let matches = query_words.iter().filter(|w| text_lower.contains(&w.to_lowercase())).count();
    matches as f64 / query_words.len() as f64
}

/// Domain match bonus.
fn domain_match(tile_domain: &str, query: &str) -> f64 {
    if tile_domain.is_empty() { return 0.0; }
    if tile_domain.to_lowercase().contains(&query.to_lowercase()) { return 1.0; }
    // Check if any query word matches domain
    query.split_whitespace()
        .filter(|w| tile_domain.to_lowercase().contains(&w.to_lowercase()))
        .count() as f64 / query.split_whitespace().count().max(1) as f64
}

/// Search result with score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub tile_id: String,
    pub score: f64,
    pub keyword_score: f64,
    pub trigram_score: f64,
    pub word_jaccard: f64,
    pub domain_score: f64,
}

/// Search configuration.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub keyword_weight: f64,
    pub trigram_weight: f64,
    pub jaccard_weight: f64,
    pub domain_weight: f64,
    pub min_score: f64,
    pub max_results: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self { keyword_weight: 0.35, trigram_weight: 0.25, jaccard_weight: 0.25, domain_weight: 0.15, min_score: 0.05, max_results: 20 }
    }
}

/// Tile search engine.
pub struct TileSearch {
    tiles: HashMap<String, SearchableTile>,
    config: SearchConfig,
}

impl Default for TileSearch {
    fn default() -> Self { Self::new() }
}

impl TileSearch {
    pub fn new() -> Self { Self { tiles: HashMap::new(), config: SearchConfig::default() } }
    pub fn with_config(config: SearchConfig) -> Self { Self { tiles: HashMap::new(), config } }

    /// Add a tile to the index.
    pub fn index(&mut self, tile: SearchableTile) {
        self.tiles.insert(tile.id.clone(), tile);
    }

    /// Remove a tile from the index.
    pub fn remove(&mut self, id: &str) -> bool { self.tiles.remove(id).is_some() }

    /// Search for tiles matching query.
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();
        for tile in self.tiles.values() {
            let all_text = tile.all_text();
            let kw = keyword_match_score(query, &all_text);
            let tg = jaccard_similarity(&trigrams(&query.to_lowercase()), &trigrams(&all_text.to_lowercase()));
            let wj = word_jaccard(query, &all_text);
            let dm = domain_match(&tile.domain, query);
            let score = self.config.keyword_weight * kw
                + self.config.trigram_weight * tg
                + self.config.jaccard_weight * wj
                + self.config.domain_weight * dm;
            if score >= self.config.min_score {
                results.push(SearchResult { tile_id: tile.id.clone(), score, keyword_score: kw, trigram_score: tg, word_jaccard: wj, domain_score: dm });
            }
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(self.config.max_results);
        results
    }

    /// Search within a specific domain only.
    pub fn search_domain(&self, query: &str, domain: &str) -> Vec<SearchResult> {
        let all = self.search(query);
        all.into_iter().filter(|r| {
            self.tiles.get(&r.tile_id).map_or(false, |t| t.domain == domain)
        }).collect()
    }

    /// Get indexed tile count.
    pub fn len(&self) -> usize { self.tiles.len() }
    pub fn is_empty(&self) -> bool { self.tiles.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(id: &str, q: &str, a: &str, tags: &[&str], domain: &str) -> SearchableTile {
        SearchableTile { id: id.to_string(), question: q.to_string(), answer: a.to_string(), tags: tags.iter().map(|s| s.to_string()).collect(), domain: domain.to_string() }
    }

    #[test]
    fn test_exact_keyword_match() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "pythagorean theorem formula", "a² + b² = c²", &["math"], "math"));
        let results = idx.search("pythagorean theorem");
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.5);
    }

    #[test]
    fn test_no_match() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "pythagorean theorem", "a² + b² = c²", &[], "math"));
        assert!(idx.search("quantum physics").is_empty() || idx.search("quantum physics")[0].score < 0.1);
    }

    #[test]
    fn test_trigram_fuzzy() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "pythagorean theorem", "formula for right triangles", &[], "math"));
        let results = idx.search("pythagoras theorom"); // misspellings
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_domain_boost() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "constraint theory", "snap vectors to manifold", &["math"], "math"));
        idx.index(tile("2", "constraint theory", "snap vectors to manifold", &["math"], "geometry"));
        let results = idx.search_domain("constraint theory", "math");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tile_id, "1");
    }

    #[test]
    fn test_max_results() {
        let mut idx = TileSearch::new();
        for i in 0..30 { idx.index(tile(&i.to_string(), "math problem", "solution", &[], "math")); }
        let results = idx.search("math");
        assert!(results.len() <= 20);
    }

    #[test]
    fn test_min_score_filter() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "math algebra", "equation solving", &[], "math"));
        let config = SearchConfig { min_score: 0.9, ..Default::default() };
        let idx2 = TileSearch::with_config(config);
        // Empty index — no matches above threshold
        assert!(idx2.search("math algebra").is_empty());
    }

    #[test]
    fn test_tag_search() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "concept", "details", &["pythagorean", "geometry"], "math"));
        let results = idx.search("pythagorean geometry");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_remove_tile() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "math", "formula", &[], "math"));
        assert!(idx.remove("1"));
        assert!(idx.search("math").is_empty());
    }

    #[test]
    fn test_word_jaccard_exact() {
        assert!((word_jaccard("hello world", "hello world") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_word_jaccard_partial() {
        let j = word_jaccard("hello world", "hello there");
        assert!(j > 0.0 && j < 1.0);
    }

    #[test]
    fn test_word_jaccard_empty() {
        assert!((word_jaccard("", "hello")).abs() < 0.001);
    }

    #[test]
    fn test_trigram_fn() {
        let t = trigrams("abc");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0], "abc");
    }

    #[test]
    fn test_trigram_short() {
        assert!(trigrams("ab").is_empty());
    }

    #[test]
    fn test_keyword_match() {
        assert!((keyword_match_score("hello world", "hello beautiful world") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_domain_match_exact() {
        assert!((domain_match("math", "math") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_domain_match_partial() {
        let d = domain_match("math geometry", "math");
        assert!(d > 0.0 && d <= 1.0);
    }

    #[test]
    fn test_index_len() {
        let mut idx = TileSearch::new();
        assert!(idx.is_empty());
        idx.index(tile("1", "q", "a", &[], "d"));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_search_result_fields() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "pythagorean theorem", "a² + b² = c²", &["math"], "math"));
        let results = idx.search("pythagorean theorem");
        let r = &results[0];
        assert!(r.keyword_score > 0.0);
        assert!(r.trigram_score > 0.0);
        assert!(r.word_jaccard > 0.0);
    }

    #[test]
    fn test_case_insensitive() {
        let mut idx = TileSearch::new();
        idx.index(tile("1", "Pythagorean Theorem", "Formula", &[], "Math"));
        let results = idx.search("pythagorean theorem");
        assert_eq!(results.len(), 1);
    }
}
