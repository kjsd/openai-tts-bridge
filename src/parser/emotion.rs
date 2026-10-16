use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Regex matching `<SE>` and `</SE>` wrapper tags used by Higgs/OpenAI audio prompts
static SE_WRAPPER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)</?se>")
        .expect("Failed to compile SE_WRAPPER_REGEX")
});

/// Regex matching full bracketed or starred tags:
/// [tag], 【tag】, (tag), （tag）, *tag*
static FULL_TAG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(?P<tag>[^\]]+)\]|【(?P<ztag>[^】]+)】|\((?P<ptag>[^)]+)\)|（(?P<fptag>[^）]+)）|\*(?P<atag>[^*]+)\*")
        .expect("Failed to compile FULL_TAG_REGEX")
});

/// Missing opening bracket: e.g. "chuckle] Hello", " clear throat] text"
/// Captures boundary prefix, ASCII tag words, and closing bracket ']'.
static MISSING_OPEN_BRACKET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?P<prefix>^|[\s.,!?。、！？])(?P<otag>[a-z][a-z0-9_\- ]{1,25})\]")
        .expect("Failed to compile MISSING_OPEN_BRACKET_REGEX")
});

/// Missing closing bracket: e.g. "[whisper Hello", " [sigh text"
/// Captures opening bracket '[', ASCII tag words, and boundary suffix.
static MISSING_CLOSE_BRACKET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\[(?P<itag>[a-z][a-z0-9_\- ]{1,25})(?P<suffix>[\s.,!?。、！？]|$)")
        .expect("Failed to compile MISSING_CLOSE_BRACKET_REGEX")
});

/// Cleans up orphan bracket or asterisk symbols from text boundaries
static ORPHAN_BRACKETS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[\[\]【】()（）*]+|[\[\]【】()（）*]+$")
        .expect("Failed to compile ORPHAN_BRACKETS_REGEX")
});

static MULTI_SPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[ \t]{2,}")
        .expect("Failed to compile MULTI_SPACE_REGEX")
});

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedPrompt {
    pub cleaned_text: String,
    pub emotion_sliders: HashMap<String, f32>,
    pub emotion_cfg_scale: f32,
    pub speed_factor: f32,
    pub detected_tags: Vec<String>,
}

impl Default for ParsedPrompt {
    fn default() -> Self {
        Self {
            cleaned_text: String::new(),
            emotion_sliders: HashMap::new(),
            emotion_cfg_scale: 1.0,
            speed_factor: 1.0,
            detected_tags: Vec::new(),
        }
    }
}

pub struct EmotionParser;

impl EmotionParser {
    pub fn parse(input: &str) -> ParsedPrompt {
        let mut detected_tags = Vec::new();
        let mut emotion_sliders = HashMap::new();
        let mut emotion_cfg_scale = 1.0f32;
        let mut speed_factor = 1.0f32;

        // 0. Strip <SE> and </SE> wrappers if present
        let mut working_text = SE_WRAPPER_REGEX.replace_all(input, " ").to_string();

        // 1. Capture and strip full enclosed tags: [...], 【...】, (...), （...）, *...*
        for cap in FULL_TAG_REGEX.captures_iter(&working_text) {
            let raw_tag = cap
                .name("tag")
                .or_else(|| cap.name("ztag"))
                .or_else(|| cap.name("ptag"))
                .or_else(|| cap.name("fptag"))
                .or_else(|| cap.name("atag"))
                .map(|m| m.as_str().trim().to_lowercase())
                .unwrap_or_default();

            if !raw_tag.is_empty() {
                detected_tags.push(raw_tag);
            }
        }
        working_text = FULL_TAG_REGEX.replace_all(&working_text, " ").to_string();

        // 2. Rescue and strip tags with missing opening bracket (e.g. "chuckle] Hello")
        for cap in MISSING_OPEN_BRACKET_REGEX.captures_iter(&working_text) {
            if let Some(m) = cap.name("otag") {
                let raw_tag = m.as_str().trim().to_lowercase();
                if !raw_tag.is_empty() {
                    detected_tags.push(raw_tag);
                }
            }
        }
        working_text = MISSING_OPEN_BRACKET_REGEX.replace_all(&working_text, "$prefix ").to_string();

        // 3. Rescue and strip tags with missing closing bracket (e.g. "[whisper Hello")
        for cap in MISSING_CLOSE_BRACKET_REGEX.captures_iter(&working_text) {
            if let Some(m) = cap.name("itag") {
                let raw_tag = m.as_str().trim().to_lowercase();
                if !raw_tag.is_empty() {
                    detected_tags.push(raw_tag);
                }
            }
        }
        working_text = MISSING_CLOSE_BRACKET_REGEX.replace_all(&working_text, " $suffix").to_string();

        // 4. Sanitize whitespace and orphan boundary bracket remnants
        let cleaned_collapsed = MULTI_SPACE_REGEX.replace_all(&working_text, " ").to_string();
        let trimmed = cleaned_collapsed.trim();
        let sanitized = ORPHAN_BRACKETS_REGEX.replace_all(trimmed, "").to_string();
        let cleaned_trimmed = sanitized.trim().to_string();

        // Apply rules based on detected tags (SkyrimNet vbeta27 carry_tags + Chatterbox + Nina custom tags)
        for tag in &detected_tags {
            match tag.as_str() {
                // Happy / Laugh / Joy / Proud / Confident
                "laughter" | "laugh" | "laughs" | "laughing" | "happy" | "happily" | "joy"
                | "joyful" | "joyfully" | "cheer" | "cheerful" | "cheerfully" | "smile"
                | "hopeful" | "proud" | "proudly" | "confident" | "confidently"
                | "emotion-happy" | "emotion-joy" | "emotion-cheerful" | "emotion-proud"
                | "sfx-laughter" | "sfx-laugh"
                | "笑" | "笑い" | "喜" | "歓喜" => {
                    emotion_sliders.insert("happy".to_string(), 0.7);
                    emotion_cfg_scale = 1.2;
                }
                "chuckle" | "chuckles" | "chuckling" | "giggle" | "giggles" | "giggling"
                | "playful" | "playfully" | "teasing" | "teasingly"
                | "flirtatious" | "flirtatiously" | "sfx-chuckle" | "sfx-giggle"
                | "style-chuckle" | "style-giggle" | "style-playful" | "style-teasing"
                | "くすくす" => {
                    emotion_sliders.insert("happy".to_string(), 0.5);
                    emotion_cfg_scale = 1.15;
                }
                "excited" | "excitedly" | "emotion-excited" | "興奮" | "わくわく" => {
                    emotion_sliders.insert("happy".to_string(), 0.8);
                    emotion_cfg_scale = 1.2;
                    speed_factor = 1.10;
                }

                // Whisper / Soft / Quiet / Gentle / Tender / Sultry (deliberate & slower pacing)
                "whisper" | "whispering" | "whispers" | "whispered" | "hushed"
                | "gentle" | "gently" | "sweet" | "soft" | "softly" | "tender" | "tenderly"
                | "warm" | "warmly" | "seductive" | "seductively" | "sultry" | "reverent"
                | "shush" | "shh" | "quiet" | "quietly" | "murmurs" | "murmuring"
                | "mutters" | "muttering" | "mumbles" | "mumbling" | "conspiratorial"
                | "style-whisper"
                | "囁き" | "ささやき" | "静かに" => {
                    emotion_sliders.insert("happy".to_string(), 0.2);
                    emotion_cfg_scale = 1.1;
                    speed_factor = 0.88;
                }

                // Sad / Cry / Sigh / Tired / Weary / Sorrow (slower & heavy pacing)
                "sigh" | "sighs" | "sighing" | "tired" | "exhausted" | "weary" | "wearily"
                | "sleepy" | "sleepily" | "resigned" | "bored" | "sfx-sigh"
                | "ため息" => {
                    emotion_sliders.insert("sad".to_string(), 0.6);
                    emotion_cfg_scale = 1.15;
                    speed_factor = 0.90;
                }
                "sad" | "sadly" | "sorrow" | "sorrowful" | "mournful" | "grief" | "grieving"
                | "melancholy" | "wistful" | "tearful" | "tearfully" | "ashamed"
                | "crying" | "cry" | "cries" | "weep" | "weeping" | "sob" | "sobs" | "sobbing"
                | "plead" | "pleading" | "emotion-sad" | "emotion-sorrow" | "emotion-melancholy"
                | "sfx-cry" | "sfx-sob"
                | "悲しい" | "哀" | "泣き" | "泣" => {
                    emotion_sliders.insert("sad".to_string(), 0.7);
                    emotion_cfg_scale = 1.2;
                    speed_factor = 0.90;
                }
                "groan" | "groans" | "groaning" | "moan" | "moans" | "moaning"
                | "strained" | "pained" | "pain" | "hoarse" | "hoarsely" | "weak" | "weakly"
                | "drunk" | "drunkenly" | "sfx-groan" | "sfx-moan" | "emotion-pain"
                | "うめき" | "呻き" => {
                    emotion_sliders.insert("sad".to_string(), 0.5);
                    emotion_sliders.insert("angry".to_string(), 0.3);
                    emotion_cfg_scale = 1.15;
                    speed_factor = 0.90;
                }
                "sniff" | "sniffs" | "sniffling" | "sfx-sniff" | "鼻をすする" => {
                    emotion_sliders.insert("sad".to_string(), 0.5);
                    emotion_cfg_scale = 1.15;
                    speed_factor = 0.92;
                }

                // Angry / Shout / Irritated / Stern / Menacing (fast & intense pacing)
                "angry" | "angrily" | "anger" | "mad" | "furious" | "furiously"
                | "irritated" | "annoyed" | "frustrated" | "impatient" | "impatiently"
                | "bitter" | "bitterly" | "shout" | "shouts" | "shouting"
                | "yell" | "yells" | "yelling" | "loudly" | "scream" | "screams" | "screaming"
                | "menacing" | "menacingly" | "threatening" | "defiant" | "defiantly"
                | "stern" | "sternly" | "commanding" | "authoritative" | "gruff" | "gruffly"
                | "emotion-angry" | "emotion-furious" | "emotion-frustrated" | "style-shout"
                | "怒り" | "怒" | "叫び" => {
                    emotion_sliders.insert("angry".to_string(), 0.75);
                    emotion_cfg_scale = 1.2;
                    speed_factor = 1.15;
                }

                // Surprise / Shock / Gasp / Curious
                "surprise" | "surprised" | "shock" | "shocked" | "curious" | "curiously"
                | "emotion-surprise" | "emotion-surprised" | "emotion-curious"
                | "驚き" | "驚" => {
                    emotion_sliders.insert("surprised".to_string(), 0.7);
                    emotion_cfg_scale = 1.2;
                }
                "gasp" | "gasps" | "gasping" | "pant" | "pants" | "panting"
                | "breathless" | "breathlessly" | "sfx-gasp" | "sfx-pant"
                | "ハッ" | "息をのむ" => {
                    emotion_sliders.insert("surprised".to_string(), 0.6);
                    emotion_sliders.insert("fear".to_string(), 0.3);
                    emotion_cfg_scale = 1.15;
                }

                // Fear / Scared / Nervous / Urgent / Trembling
                "fear" | "fearful" | "scared" | "terrified" | "frightened" | "afraid" | "panicked"
                | "nervous" | "nervously" | "anxious" | "anxiously" | "worried"
                | "trembling" | "shaky" | "shakily" | "hesitant" | "hesitantly"
                | "shy" | "shyly" | "timid" | "timidly" | "embarrassed"
                | "urgent" | "urgently" | "desperate" | "desperately"
                | "emotion-fear" | "emotion-anxious" | "emotion-nervous" | "emotion-panicked"
                | "恐れ" | "怖" | "恐怖" => {
                    emotion_sliders.insert("fear".to_string(), 0.7);
                    emotion_cfg_scale = 1.2;
                }

                // Disgust
                "disgust" | "disgusted" | "emotion-disgust" | "嫌悪" => {
                    emotion_sliders.insert("disgust".to_string(), 0.7);
                    emotion_cfg_scale = 1.2;
                }

                // Sarcastic / Mocking / Smug / Cold / Dismissive
                "sarcastic" | "sarcastically" | "sarcasm" | "ironic"
                | "mocking" | "mockingly" | "smug" | "smugly"
                | "cold" | "coldly" | "grim" | "grimly" | "dismissive" | "dismissively"
                | "suspicious" | "suspiciously" | "style-sarcastic" | "style-mocking"
                | "皮肉" => {
                    emotion_sliders.insert("angry".to_string(), 0.3);
                    emotion_sliders.insert("sad".to_string(), 0.2);
                    emotion_cfg_scale = 1.15;
                    speed_factor = 0.95;
                }
                "dramatic" | "dramatic tone" | "solemn" | "solemnly" => {
                    emotion_sliders.insert("surprised".to_string(), 0.3);
                    emotion_sliders.insert("sad".to_string(), 0.2);
                    emotion_cfg_scale = 1.2;
                    speed_factor = 0.90;
                }
                "advertisement" | "commercial" | "sings" | "singing" | "style-singing" => {
                    emotion_sliders.insert("happy".to_string(), 0.4);
                    emotion_cfg_scale = 1.15;
                    speed_factor = 1.05;
                }
                "narration" | "narrator" | "calm" | "calmly" | "serious" | "seriously"
                | "thoughtful" | "thoughtfully" | "determined" | "relieved"
                | "deadpan" | "monotone" | "confused"
                | "emotion-calm" | "emotion-relieved" | "style-narration" => {
                    // Calm and neutral reading
                    emotion_cfg_scale = 1.0;
                    speed_factor = 1.00;
                }
                "slowly" | "slow" | "deliberate" | "deliberately" | "prosody-slow" => {
                    // Explicit slow pacing requested by prompt
                    speed_factor = 0.75;
                }
                "quickly" | "quick" | "fast" | "rushed" | "hurried" | "hurriedly"
                | "prosody-fast" => {
                    // Explicit fast pacing requested by prompt
                    speed_factor = 1.20;
                }

                // Pure sound/action tags (safely cleaned from text without altering emotion)
                "clear throat" | "clears throat" | "cough" | "coughs" | "pause"
                | "sneeze" | "sneezes" | "yawn" | "yawns" | "gulp" | "swallow" => {
                    // Do not alter emotions; these are stripped from text so they won't be spoken aloud
                }

                _ => {
                    // Unknown or unsupported tag, safely stripped from text
                }
            }
        }

        ParsedPrompt {
            cleaned_text: cleaned_trimmed,
            emotion_sliders,
            emotion_cfg_scale,
            speed_factor,
            detected_tags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_whisper() {
        let input = "[whisper] パパ、寒くない…？";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "パパ、寒くない…？");
        assert_eq!(parsed.detected_tags, vec!["whisper"]);
        assert_eq!(parsed.speed_factor, 0.88);
        assert_eq!(parsed.emotion_cfg_scale, 1.1);
        assert_eq!(parsed.emotion_sliders.get("happy"), Some(&0.2));
    }

    #[test]
    fn test_parse_laughter() {
        let input = "[laughter] えへへ、パパ大好き！";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "えへへ、パパ大好き！");
        assert_eq!(parsed.detected_tags, vec!["laughter"]);
        assert_eq!(parsed.emotion_cfg_scale, 1.2);
        assert_eq!(parsed.emotion_sliders.get("happy"), Some(&0.7));
    }

    #[test]
    fn test_parse_japanese_bracket() {
        let input = "【笑い】パパ、これ見て！";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "パパ、これ見て！");
        assert_eq!(parsed.detected_tags, vec!["笑い"]);
        assert_eq!(parsed.emotion_sliders.get("happy"), Some(&0.7));
    }

    #[test]
    fn test_parse_plain_text() {
        let input = "普通のテキストです。";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "普通のテキストです。");
        assert!(parsed.detected_tags.is_empty());
        assert_eq!(parsed.speed_factor, 1.0);
        assert_eq!(parsed.emotion_cfg_scale, 1.0);
        assert!(parsed.emotion_sliders.is_empty());
    }

    #[test]
    fn test_parse_skyrimnet_chatterbox_tags() {
        // Multi-word tag with space, sound tags in middle of sentence
        let input = "[whispering] Keep your voice down - [shush] - they’ll hear us. [sigh]";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "Keep your voice down - - they’ll hear us.");
        assert_eq!(parsed.detected_tags, vec!["whispering", "shush", "sigh"]);

        let input2 = "[happy] [clear throat] So… we agree? [chuckle] Great.";
        let parsed2 = EmotionParser::parse(input2);
        assert_eq!(parsed2.cleaned_text, "So… we agree? Great.");
        assert_eq!(parsed2.detected_tags, vec!["happy", "clear throat", "chuckle"]);
        assert_eq!(parsed2.emotion_sliders.get("happy"), Some(&0.5)); // chuckle overwrote happy to 0.5
    }

    #[test]
    fn test_parse_sound_tags_stripping() {
        let input = "I told you not to touch that - [groan] - ever again. [cough]";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "I told you not to touch that - - ever again.");
        assert_eq!(parsed.detected_tags, vec!["groan", "cough"]);
    }

    #[test]
    fn test_parse_dramatic_tone_and_sarcastic() {
        let input = "[dramatic tone] The dragons have returned to Skyrim.";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "The dragons have returned to Skyrim.");
        assert_eq!(parsed.detected_tags, vec!["dramatic tone"]);
        assert_eq!(parsed.emotion_cfg_scale, 1.2);
        assert_eq!(parsed.speed_factor, 0.90);

        let input_sarcastic = "[sarcastic] Oh, what an amazing hero you are.";
        let parsed_sarcastic = EmotionParser::parse(input_sarcastic);
        assert_eq!(parsed_sarcastic.cleaned_text, "Oh, what an amazing hero you are.");
        assert_eq!(parsed_sarcastic.detected_tags, vec!["sarcastic"]);
    }

    #[test]
    fn test_parse_slowly_tag() {
        let input = "[slowly] Take your time, traveler.";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "Take your time, traveler.");
        assert_eq!(parsed.detected_tags, vec!["slowly"]);
        assert_eq!(parsed.speed_factor, 0.75);
    }

    #[test]
    fn test_parse_missing_opening_bracket_skyrimnet_anomaly() {
        // Cases directly observed in SkyrimNet split logs where '[' was truncated
        let input1 = "chuckle] さあケンジ君、やっとウィンターホールドに着いたわ。";
        let parsed1 = EmotionParser::parse(input1);
        assert_eq!(parsed1.cleaned_text, "さあケンジ君、やっとウィンターホールドに着いたわ。");
        assert_eq!(parsed1.detected_tags, vec!["chuckle"]);
        assert_eq!(parsed1.emotion_sliders.get("happy"), Some(&0.5));

        let input2 = "clear throat] でも、馬車の中なら変な輩に邪魔される心配もないわね。";
        let parsed2 = EmotionParser::parse(input2);
        assert_eq!(parsed2.cleaned_text, "でも、馬車の中なら変な輩に邪魔される心配もないわね。");
        assert_eq!(parsed2.detected_tags, vec!["clear throat"]);

        let input3 = "shush] 攻撃をやめなさい！";
        let parsed3 = EmotionParser::parse(input3);
        assert_eq!(parsed3.cleaned_text, "攻撃をやめなさい！");
        assert_eq!(parsed3.detected_tags, vec!["shush"]);

        let input4 = "chuckle] [whispering] 分かったよ、イリア。";
        let parsed4 = EmotionParser::parse(input4);
        assert_eq!(parsed4.cleaned_text, "分かったよ、イリア。");
        assert_eq!(parsed4.detected_tags, vec!["whispering", "chuckle"]);
    }

    #[test]
    fn test_parse_missing_closing_bracket() {
        let input = "[whisper パパ、寒くない？";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "パパ、寒くない？");
        assert_eq!(parsed.detected_tags, vec!["whisper"]);
    }

    #[test]
    fn test_parse_parentheses_and_asterisks() {
        let input1 = "(sigh) イリアさん、どうかその魔法を収めて！";
        let parsed1 = EmotionParser::parse(input1);
        assert_eq!(parsed1.cleaned_text, "イリアさん、どうかその魔法を収めて！");
        assert_eq!(parsed1.detected_tags, vec!["sigh"]);

        let input2 = "*chuckle* おやすみなさい。";
        let parsed2 = EmotionParser::parse(input2);
        assert_eq!(parsed2.cleaned_text, "おやすみなさい。");
        assert_eq!(parsed2.detected_tags, vec!["chuckle"]);
    }
}
