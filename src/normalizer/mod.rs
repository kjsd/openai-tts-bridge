use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictionaryEntry {
    pub surface: String,
    pub reading: String,
    #[serde(default)]
    pub case_sensitive: bool,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EngineNormFlags {
    pub use_dictionary: bool,
    pub fix_wave_dash: bool,
    pub normalize_punctuation: bool,
    pub convert_units: bool,
}

#[derive(Debug, Clone)]
struct CompiledRule {
    surface: String,
    surface_lower: String,
    reading: String,
    case_sensitive: bool,
    starts_ascii_alnum: bool,
    ends_ascii_alnum: bool,
    char_len: usize,
}

#[derive(Debug, Clone, Default)]
pub struct CompiledDictionary {
    rules: Vec<CompiledRule>,
}

impl CompiledDictionary {
    pub fn compile(entries: &[DictionaryEntry]) -> Self {
        let mut rules: Vec<CompiledRule> = entries
            .iter()
            .filter(|e| e.enabled && !e.surface.trim().is_empty())
            .map(|e| {
                let surface = e.surface.trim().to_string();
                let starts_ascii_alnum = surface
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_alphanumeric())
                    .unwrap_or(false);
                let ends_ascii_alnum = surface
                    .chars()
                    .last()
                    .map(|c| c.is_ascii_alphanumeric())
                    .unwrap_or(false);
                let char_len = surface.chars().count();
                CompiledRule {
                    surface_lower: surface.to_ascii_lowercase(),
                    surface,
                    reading: e.reading.clone(),
                    case_sensitive: e.case_sensitive,
                    starts_ascii_alnum,
                    ends_ascii_alnum,
                    char_len,
                }
            })
            .collect();

        // Longest surface first (e.g. "東白川村" before "東白川", "mBot2" before "mBot")
        rules.sort_by(|a, b| b.char_len.cmp(&a.char_len));
        Self { rules }
    }

    /// Single-pass left-to-right longest-match dictionary substitution.
    /// Enforces ASCII alphanumeric boundaries for ASCII terms so "AI" matches "AIが"
    /// without corrupting English words like "wait".
    pub fn apply(&self, input: &str) -> String {
        if self.rules.is_empty() || input.is_empty() {
            return input.to_string();
        }

        let chars: Vec<(usize, char)> = input.char_indices().collect();
        let total = chars.len();
        let mut out = String::with_capacity(input.len());
        let mut idx = 0;

        while idx < total {
            let prev_char = if idx > 0 { Some(chars[idx - 1].1) } else { None };
            let mut matched: Option<&CompiledRule> = None;

            for rule in &self.rules {
                if idx + rule.char_len > total {
                    continue;
                }

                if rule.starts_ascii_alnum
                    && prev_char.map(|c| c.is_ascii_alphanumeric()).unwrap_or(false)
                {
                    continue;
                }

                let end_idx = idx + rule.char_len;
                let next_char = if end_idx < total {
                    Some(chars[end_idx].1)
                } else {
                    None
                };

                if rule.ends_ascii_alnum
                    && next_char.map(|c| c.is_ascii_alphanumeric()).unwrap_or(false)
                {
                    continue;
                }

                let byte_start = chars[idx].0;
                let byte_end = if end_idx < total {
                    chars[end_idx].0
                } else {
                    input.len()
                };
                let slice = &input[byte_start..byte_end];

                let is_match = if rule.case_sensitive {
                    slice == rule.surface
                } else {
                    slice.eq_ignore_ascii_case(&rule.surface) || slice.to_lowercase() == rule.surface_lower
                };

                if is_match {
                    matched = Some(rule);
                    break;
                }
            }

            if let Some(rule) = matched {
                out.push_str(&rule.reading);
                idx += rule.char_len;
            } else {
                out.push(chars[idx].1);
                idx += 1;
            }
        }

        out
    }
}

static FENCED_CODE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)```.*?```").unwrap());
static INLINE_CODE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([^`]+)`").unwrap());
static MD_IMAGE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!\[[^\]]*\]\([^)]+\)").unwrap());
static MD_LINK_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\((?:https?://|/|#)[^)]+\)").unwrap());
static MD_BOLD_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\*{2,}([^*]+)\*{2,}|_{2,}([^_]+)_{2,}").unwrap());

/// Strips non-spoken Markdown formatting while preserving `[tag]` and single `*tag*` audio annotations.
pub fn strip_markdown(text: &str) -> String {
    let s = FENCED_CODE_REGEX.replace_all(text, "");
    let s = INLINE_CODE_REGEX.replace_all(&s, "$1");
    let s = MD_IMAGE_REGEX.replace_all(&s, "");
    let s = MD_LINK_REGEX.replace_all(&s, "$1");
    let s = MD_BOLD_REGEX.replace_all(&s, |caps: &regex::Captures| {
        caps.get(1)
            .or_else(|| caps.get(2))
            .map(|m| m.as_str())
            .unwrap_or("")
            .to_string()
    });
    s.into_owned()
}

static UNIT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>km/h|cm/s|m/s|cm|mm|km|kg|deg|℃|°c|°|[%％]|m|g)",
    )
    .unwrap()
});

/// Converts numeric units (e.g. `10cm`, `50%`, `30km/h`, `25℃`) into natural Japanese readings,
/// ensuring ASCII unit suffixes are not prefixes of English words (like `10 minutes`).
pub fn convert_units(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_end = 0;

    for caps in UNIT_REGEX.captures_iter(text) {
        let m = caps.get(0).unwrap();
        let after = &text[m.end()..];
        let unit_raw = caps.name("unit").unwrap().as_str();

        // If the unit ends with an ASCII letter, ensure the next char is not an ASCII letter
        let ends_ascii_alpha = unit_raw
            .chars()
            .last()
            .map(|c| c.is_ascii_alphabetic())
            .unwrap_or(false);
        if ends_ascii_alpha
            && after
                .chars()
                .next()
                .map(|c| c.is_ascii_alphabetic())
                .unwrap_or(false)
        {
            continue;
        }

        out.push_str(&text[last_end..m.start()]);
        let num = caps.name("num").unwrap().as_str();
        let unit_lower = unit_raw.to_ascii_lowercase();

        match unit_lower.as_str() {
            "km/h" => {
                out.push_str(num);
                out.push_str("キロ");
            }
            "cm/s" => {
                out.push_str("秒速");
                out.push_str(num);
                out.push_str("センチ");
            }
            "m/s" => {
                out.push_str("秒速");
                out.push_str(num);
                out.push_str("メートル");
            }
            "cm" => {
                out.push_str(num);
                out.push_str("センチ");
            }
            "mm" => {
                out.push_str(num);
                out.push_str("ミリ");
            }
            "km" => {
                out.push_str(num);
                out.push_str("キロ");
            }
            "m" => {
                out.push_str(num);
                out.push_str("メートル");
            }
            "kg" => {
                out.push_str(num);
                out.push_str("キログラム");
            }
            "g" => {
                out.push_str(num);
                out.push_str("グラム");
            }
            "deg" | "℃" | "°c" | "°" => {
                out.push_str(num);
                out.push_str("度");
            }
            "%" | "％" => {
                out.push_str(num);
                out.push_str("パーセント");
            }
            _ => {
                out.push_str(m.as_str());
            }
        }

        last_end = m.end();
    }

    out.push_str(&text[last_end..]);
    out
}

static RANGE_WAVE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?P<lhs>[0-9０-９]+(?:\s*(?:時|分|秒|年|月|日|歳|才|回|個|人|つ|番|階|章|ページ|センチ|メートル|キロ|度|パーセント))?)\s*[〜～]+\s*(?P<rhs>[0-9０-９]+)",
    )
    .unwrap()
});
static REMAINING_WAVE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[〜～]+").unwrap());

/// Context-aware wave dash (`〜` / `～`) normalization:
/// - Numeric/time ranges like `1〜2個` or `10時〜12時` become `1から2個` / `10時から12時`.
/// - Expressive elongation like `の〜んびり` or `嬉しいな〜` becomes `のーんびり` / `嬉しいなー`
///   so Zonos 2 never misreads `〜` as `から`.
pub fn fix_wave_dash(text: &str) -> String {
    let with_ranges = RANGE_WAVE_REGEX.replace_all(text, "${lhs}から${rhs}");
    REMAINING_WAVE_REGEX
        .replace_all(&with_ranges, "ー")
        .into_owned()
}

static ELLIPSIS_BEFORE_PUNCT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:…|\.{2,}|[、,，\s])+([。！？!?]+)").unwrap());
static TRAILING_ELLIPSIS_END_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:…|\.{3,}|[、,，\s])+$").unwrap());
static MID_ELLIPSIS_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:…+|\.{3,})").unwrap());
static DUP_COMMA_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"、{2,}").unwrap());

/// Normalizes ellipses and trailing loose punctuation for Zonos 2 / Zonos v0.1
/// so phonemizers pause naturally mid-sentence (`、` instead of `…`) and do not hallucinate at sentence boundaries.
pub fn normalize_zonos_punctuation(text: &str) -> String {
    let s = ELLIPSIS_BEFORE_PUNCT_REGEX.replace_all(text.trim(), "$1");
    let s = TRAILING_ELLIPSIS_END_REGEX.replace_all(&s, "");
    let s = MID_ELLIPSIS_REGEX.replace_all(&s, "、");
    let s = DUP_COMMA_REGEX.replace_all(&s, "、");
    // Second pass in case MID_ELLIPSIS_REGEX produced `、！`
    let s = ELLIPSIS_BEFORE_PUNCT_REGEX.replace_all(&s, "$1");
    s.trim().to_string()
}

/// Applies the configured per-engine normalization steps to tag-stripped (or emoji-mapped) text.
pub fn apply_engine_normalization(
    text: &str,
    dict: &CompiledDictionary,
    flags: EngineNormFlags,
) -> String {
    let mut s = text.to_string();
    if flags.convert_units {
        s = convert_units(&s);
    }
    if flags.use_dictionary {
        s = dict.apply(&s);
    }
    if flags.fix_wave_dash {
        s = fix_wave_dash(&s);
    }
    if flags.normalize_punctuation {
        s = normalize_zonos_punctuation(&s);
    }
    s
}

static SPEAKABLE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{Hiragana}\p{Katakana}\p{Han}a-zA-Z0-9]").unwrap()
});
static SENTENCE_SPLIT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([。！？!?]+|\r?\n+)").unwrap());
static ZONOS_V1_TRAIL_CLEAN_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[、,，…〜～\.・\s]+([。！？]+)$").unwrap());
static ZONOS_V1_END_STRIP_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[、,，…〜～\.・\s]+$").unwrap());

pub fn is_speakable(text: &str) -> bool {
    SPEAKABLE_REGEX.is_match(text)
}

/// Splits normalized text into clean sentence chunks for Zonos v0.1 (`espeak-ng`),
/// ensuring each chunk terminates with explicit Japanese terminal punctuation (`。`, `！`, `？`).
pub fn split_zonos_v1_chunks(text: &str) -> Vec<String> {
    let mut raw_sentences: Vec<String> = Vec::new();
    let mut last = 0;

    for m in SENTENCE_SPLIT_REGEX.find_iter(text) {
        let seg = text[last..m.start()].trim();
        let delim = m.as_str().trim();
        if !seg.is_empty() {
            let mut combined = seg.to_string();
            if !delim.is_empty() {
                combined.push_str(delim);
            }
            raw_sentences.push(combined);
        } else if !delim.is_empty() {
            if let Some(prev) = raw_sentences.last_mut() {
                prev.push_str(delim);
            }
        }
        last = m.end();
    }

    let tail = text[last..].trim();
    if !tail.is_empty() {
        raw_sentences.push(tail.to_string());
    }

    let mut out = Vec::new();
    for sent in raw_sentences {
        if !is_speakable(&sent) {
            continue;
        }
        let s = sent.replace('!', "！").replace('?', "？");
        let s = ZONOS_V1_TRAIL_CLEAN_REGEX.replace_all(&s, "$1");
        let s = ZONOS_V1_END_STRIP_REGEX.replace_all(&s, "");
        let s = MID_ELLIPSIS_REGEX.replace_all(&s, "、");
        let s = DUP_COMMA_REGEX.replace_all(&s, "、");
        let cleaned = s.trim();
        if !is_speakable(cleaned) {
            continue;
        }
        if cleaned.ends_with('。') || cleaned.ends_with('！') || cleaned.ends_with('？') {
            out.push(cleaned.to_string());
        } else {
            out.push(format!("{cleaned}。"));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_longest_match_and_ascii_boundaries() {
        let dict = CompiledDictionary::compile(&[
            DictionaryEntry {
                surface: "東白川".to_string(),
                reading: "ひがししらかわ".to_string(),
                case_sensitive: true,
                enabled: true,
            },
            DictionaryEntry {
                surface: "東白川村".to_string(),
                reading: "ひがししらかわむら".to_string(),
                case_sensitive: true,
                enabled: true,
            },
            DictionaryEntry {
                surface: "mBot".to_string(),
                reading: "エムボット".to_string(),
                case_sensitive: false,
                enabled: true,
            },
            DictionaryEntry {
                surface: "mBot2".to_string(),
                reading: "エムボットツー".to_string(),
                case_sensitive: false,
                enabled: true,
            },
            DictionaryEntry {
                surface: "AI".to_string(),
                reading: "エーアイ".to_string(),
                case_sensitive: false,
                enabled: true,
            },
        ]);

        assert_eq!(
            dict.apply("東白川村でmbot2とAIを使う"),
            "ひがししらかわむらでエムボットツーとエーアイを使う"
        );
        // Must NOT corrupt "wait" even though "AI" is in dictionary
        assert_eq!(dict.apply("Please wait for AI!"), "Please wait for エーアイ!");
    }

    #[test]
    fn test_fix_wave_dash_context_aware() {
        assert_eq!(
            fix_wave_dash("の〜んびりしようね〜！"),
            "のーんびりしようねー！"
        );
        assert_eq!(
            fix_wave_dash("1〜2個と10時～12時の予定"),
            "1から2個と10時から12時の予定"
        );
    }

    #[test]
    fn test_convert_units_with_japanese_and_english() {
        assert_eq!(
            convert_units("10cm前進して50%の出力で30km/h、気温25℃"),
            "10センチ前進して50パーセントの出力で30キロ、気温25度"
        );
        // Must NOT turn "10 minutes" into "10メートルinutes"
        assert_eq!(convert_units("Wait 10 minutes"), "Wait 10 minutes");
    }

    #[test]
    fn test_zonos_v1_sentence_chunk_splitting() {
        let chunks = split_zonos_v1_chunks("こんにちは…！ 今日もいい天気だね〜\nよし行こう");
        assert_eq!(
            chunks,
            vec!["こんにちは！", "今日もいい天気だね。", "よし行こう。"]
        );
        assert_eq!(
            normalize_zonos_punctuation("東白川村へ行こう…！ 次はそっち…かな…"),
            "東白川村へ行こう！ 次はそっち、かな"
        );
    }

    #[test]
    fn test_strip_markdown_preserves_audio_tags() {
        let raw = "[whisper] **大好き**だよ *chuckle* `code` ![img](/a.png)";
        assert_eq!(strip_markdown(raw), "[whisper] 大好きだよ *chuckle* code ");
    }
}
