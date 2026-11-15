use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Regex matching `<SE>` and `</SE>` wrapper tags used by Higgs/OpenAI audio prompts
static SE_WRAPPER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)</?se>").expect("Failed to compile SE_WRAPPER_REGEX")
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
    Regex::new(r"(?i)(?P<prefix>^|[\s.,!?。、！？])(?P<otag>[a-z][a-z0-9_\-: ]{1,30})\]")
        .expect("Failed to compile MISSING_OPEN_BRACKET_REGEX")
});

/// Missing closing bracket: e.g. "[whisper Hello", " [sigh text"
/// Captures opening bracket '[', ASCII tag words, and boundary suffix.
static MISSING_CLOSE_BRACKET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\[(?P<itag>[a-z][a-z0-9_\-: ]{1,30})(?P<suffix>[\s.,!?。、！？]|$)")
        .expect("Failed to compile MISSING_CLOSE_BRACKET_REGEX")
});

/// Cleans up orphan bracket or asterisk symbols from text boundaries
static ORPHAN_BRACKETS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[\[\]【】()（）*]+|[\[\]【】()（）*]+$")
        .expect("Failed to compile ORPHAN_BRACKETS_REGEX")
});

static MULTI_SPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[ \t]{2,}").expect("Failed to compile MULTI_SPACE_REGEX")
});

static SPEAKABLE_CHAR_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{Hiragana}\p{Katakana}\p{Han}a-zA-Z0-9]")
        .expect("Failed to compile SPEAKABLE_CHAR_REGEX")
});

const MODIFIER_PREFIXES: &[&str] = &[
    "very ",
    "deep ",
    "deeply ",
    "heavy ",
    "heavily ",
    "extremely ",
    "intense ",
    "intensely ",
    "strong ",
    "strongly ",
    "loud ",
    "loudly ",
    "super ",
    "slightly ",
    "mildly ",
    "a bit ",
    "a little ",
    "somewhat ",
    "soft ",
    "softly ",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TagLayer {
    /// Persistent tone / emotion / style that carries across sentences until overridden or reset
    Carry,
    /// Instantaneous vocal reaction applied only to the current sentence
    OneShot,
    /// Resets any active Carry / BaseMood back to calm neutral speech
    Reset,
    /// Non-emotional sound or action tag (stripped from text without altering emotion)
    Neutral,
}

#[derive(Debug, Clone)]
pub struct EmotionProfile {
    pub layer: TagLayer,
    pub sliders: HashMap<String, f32>,
    pub valence: Option<f32>,
    pub arousal: Option<f32>,
    pub cfg_scale: f32,
    pub speed_factor: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedPrompt {
    pub cleaned_text: String,
    pub emotion_enabled: bool,
    pub emotion_sliders: HashMap<String, f32>,
    pub emotion_valence: Option<f32>,
    pub emotion_arousal: Option<f32>,
    pub emotion_cfg_scale: f32,
    pub speed_factor: f32,
    pub accurate_mode: bool,
    pub detected_tags: Vec<String>,
    #[serde(default)]
    pub effective_tags: Vec<String>,
}

impl Default for ParsedPrompt {
    fn default() -> Self {
        Self {
            cleaned_text: String::new(),
            emotion_enabled: false,
            emotion_sliders: HashMap::new(),
            emotion_valence: None,
            emotion_arousal: None,
            emotion_cfg_scale: 1.0,
            speed_factor: 1.0,
            // Always keep accurate_mode=false across neutral and emotional chunks so Zonos 2
            // uses a consistent decoding pathway and never shifts speaker timbre mid-turn.
            accurate_mode: false,
            detected_tags: Vec::new(),
            effective_tags: Vec::new(),
        }
    }
}

pub struct EmotionParser;

impl EmotionParser {
    const CARRY_BLEND_WEIGHT: f32 = 0.35;
    const ONESHOT_BLEND_WEIGHT: f32 = 0.65;
    const BASE_CARRY_WEIGHT: f32 = 0.30;
    const OVERRIDE_CARRY_WEIGHT: f32 = 0.70;

    pub fn strip_modifier_prefix(tag: &str) -> &str {
        for prefix in MODIFIER_PREFIXES {
            if let Some(rest) = tag.strip_prefix(prefix) {
                return rest.trim();
            }
        }
        tag
    }

    fn round3(v: f32) -> f32 {
        (v * 1000.0).round() / 1000.0
    }

    fn normalize_sliders(mut sliders: HashMap<String, f32>) -> HashMap<String, f32> {
        let total: f32 = sliders.values().copied().sum();
        if total > 1.0 {
            for val in sliders.values_mut() {
                *val = Self::round3(*val / total);
            }
        } else {
            for val in sliders.values_mut() {
                *val = Self::round3(*val);
            }
        }
        sliders
    }

    fn blend_weighted_profiles(weighted: &[(&EmotionProfile, f32)]) -> EmotionProfile {
        let total_w: f32 = weighted.iter().map(|(_, w)| *w).sum();
        let norm: Vec<(&EmotionProfile, f32)> = if total_w > 0.0 {
            weighted.iter().map(|(p, w)| (*p, *w / total_w)).collect()
        } else {
            weighted.to_vec()
        };

        let mut merged_sliders: HashMap<String, f32> = HashMap::new();
        for (p, w) in &norm {
            for (k, v) in &p.sliders {
                *merged_sliders.entry(k.clone()).or_insert(0.0) += *v * *w;
            }
        }
        merged_sliders.retain(|_, v| *v > 0.001);
        let sliders = Self::normalize_sliders(merged_sliders);

        // Blend valence across profiles that specify valence
        let valence = {
            let present: Vec<(f32, f32)> = norm
                .iter()
                .filter_map(|(p, w)| p.valence.map(|v| (v, *w)))
                .collect();
            if present.is_empty() {
                None
            } else {
                let w_sum: f32 = present.iter().map(|(_, w)| *w).sum();
                let val: f32 = present.iter().map(|(v, w)| v * (w / w_sum)).sum();
                Some(Self::round3(val.clamp(-1.0, 1.0)))
            }
        };

        // Blend arousal across profiles that specify arousal
        let arousal = {
            let present: Vec<(f32, f32)> = norm
                .iter()
                .filter_map(|(p, w)| p.arousal.map(|a| (a, *w)))
                .collect();
            if present.is_empty() {
                None
            } else {
                let w_sum: f32 = present.iter().map(|(_, w)| *w).sum();
                let val: f32 = present.iter().map(|(a, w)| a * (w / w_sum)).sum();
                Some(Self::round3(val.clamp(-1.0, 1.0)))
            }
        };

        let cfg_scale = Self::round3(norm.iter().map(|(p, w)| p.cfg_scale * *w).sum());
        let speed_factor = Self::round3(norm.iter().map(|(p, w)| p.speed_factor * *w).sum());
        let last_layer = norm.last().map(|(p, _)| p.layer).unwrap_or(TagLayer::Carry);

        EmotionProfile {
            layer: last_layer,
            sliders,
            valence,
            arousal,
            cfg_scale,
            speed_factor,
        }
    }

    fn blend_profiles(p1: &EmotionProfile, w1: f32, p2: &EmotionProfile, w2: f32) -> EmotionProfile {
        Self::blend_weighted_profiles(&[(p1, w1), (p2, w2)])
    }

    /// Scales down a turn-wide `[mood:<tag>]` foundation so it anchors speaker tone gently
    /// without overpowering sentence-level Carry or One-Shot tags.
    fn scale_base_mood(profile: &EmotionProfile) -> EmotionProfile {
        let mut sliders = HashMap::new();
        for (k, v) in &profile.sliders {
            let scaled = Self::round3(v * 0.75);
            if scaled > 0.001 {
                sliders.insert(k.clone(), scaled);
            }
        }
        EmotionProfile {
            layer: TagLayer::Carry,
            sliders,
            valence: profile.valence.map(|v| Self::round3(v * 0.8)),
            arousal: profile.arousal.map(|a| Self::round3(a * 0.8)),
            cfg_scale: Self::round3(1.0 + (profile.cfg_scale - 1.0) * 0.6),
            speed_factor: Self::round3(1.0 + (profile.speed_factor - 1.0) * 0.5),
        }
    }

    /// Maps a normalized base tag to its `EmotionProfile` (Carry, OneShot, Reset, or Neutral).
    /// Tuned with conservative `cfg_scale` (1.00..=1.15) and separated vocal profiles so Zonos 2
    /// maintains stable speaker identity while expressing clear emotional nuance.
    pub fn classify_tag(base_tag: &str) -> Option<EmotionProfile> {
        let mut sliders = HashMap::new();
        match base_tag {
            // 1a. One-Shot Laugh
            "laughter" | "laugh" | "laughs" | "laughing" | "cackle" | "cackles" | "cackling"
            | "chortle" | "sfx-laughter" | "sfx-laugh" | "笑" | "笑い" => {
                sliders.insert("happy".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.65),
                    arousal: Some(0.45),
                    cfg_scale: 1.12,
                    speed_factor: 1.0,
                })
            }
            // 1b. Carry Happy / Joy / Proud / Confident / Glad
            "happy" | "happily" | "joy" | "joyful" | "joyfully" | "cheer" | "cheerful"
            | "cheerfully" | "glad" | "gladly" | "delighted" | "delightedly" | "pleased"
            | "smile" | "smiling" | "hopeful" | "hopefully" | "proud" | "proudly"
            | "confident" | "confidently" | "emotion-happy" | "emotion-joy"
            | "emotion-elation" | "emotion-contentment" | "emotion-cheerful" | "emotion-proud"
            | "喜" | "喜び" | "歓喜" | "楽しげ" | "嬉しそう" => {
                sliders.insert("happy".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.55),
                    arousal: Some(0.3),
                    cfg_scale: 1.10,
                    speed_factor: 1.0,
                })
            }

            // 2a. One-Shot Chuckle / Giggle / Snicker
            "chuckle" | "chuckles" | "chuckling" | "giggle" | "giggles" | "giggling"
            | "snicker" | "snickers" | "snickering" | "titter" | "titters"
            | "sfx-chuckle" | "sfx-giggle" | "style-chuckle" | "style-giggle"
            | "くすくす" | "含み笑い" => {
                sliders.insert("happy".to_string(), 0.5);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.45),
                    arousal: Some(0.2),
                    cfg_scale: 1.08,
                    speed_factor: 1.0,
                })
            }
            // 2b. Carry Playful / Teasing / Coaxing / Flirty / Seductive
            "playful" | "playfully" | "teasing" | "teasingly" | "tease" | "coaxing"
            | "coaxingly" | "coax" | "sweetly" | "coy" | "coyly" | "flirtatious"
            | "flirtatiously" | "flirty" | "seductive" | "seductively" | "sultry"
            | "amused" | "amusedly" | "emotion-amusement" | "emotion-arousal"
            | "style-playful" | "style-teasing" | "からかい" | "甘え" => {
                sliders.insert("happy".to_string(), 0.65);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.45),
                    arousal: Some(0.25),
                    cfg_scale: 1.08,
                    speed_factor: 0.96,
                })
            }

            // 3. Carry Excited / Enthusiastic
            "excited" | "excitedly" | "gleeful" | "gleefully" | "ecstatic"
            | "enthusiastic" | "enthusiastically" | "emotion-excited" | "emotion-enthusiasm"
            | "興奮" | "わくわく" => {
                sliders.insert("happy".to_string(), 0.8);
                sliders.insert("surprised".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.65),
                    arousal: Some(0.55),
                    cfg_scale: 1.12,
                    speed_factor: 1.05,
                })
            }

            // 4a. One-Shot Shush
            "shush" | "shh" => {
                sliders.insert("happy".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.1),
                    arousal: Some(-0.4),
                    cfg_scale: 1.0,
                    speed_factor: 0.88,
                })
            }
            // 4b. Carry Whisper / Hushed / Quiet (cfg_scale=1.0 prevents whisper timbre distortion)
            "whisper" | "whispering" | "whispers" | "whispered" | "hushed"
            | "quiet" | "quietly" | "murmur" | "murmurs" | "murmuring"
            | "mutter" | "mutters" | "muttering" | "mumble" | "mumbles" | "mumbling"
            | "conspiratorial" | "conspiratorially" | "under breath" | "muffled"
            | "style-whisper" | "style-whispering"
            | "囁き" | "ささやき" | "小声" | "静かに" => {
                sliders.insert("happy".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.2),
                    arousal: Some(-0.3),
                    cfg_scale: 1.0,
                    speed_factor: 0.88,
                })
            }
            // 4c. Carry Gently / Tender / Warm / Loving (distinct from whisper!)
            "gentle" | "gently" | "sweet" | "soft" | "softly" | "tender" | "tenderly"
            | "warm" | "warmly" | "kind" | "kindly" | "affectionate" | "affectionately"
            | "loving" | "lovingly" | "caring" | "caringly" | "soothing" | "soothingly"
            | "comforting" | "comfortingly" | "reverent" | "reverently" | "emotion-affection"
            | "優しく" | "慈しむ" | "穏やか" => {
                sliders.insert("happy".to_string(), 0.5);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.45),
                    arousal: Some(-0.2),
                    cfg_scale: 1.05,
                    speed_factor: 0.92,
                })
            }
            // 4d. Carry Shy / Bashful / Blushing (sweet bashfulness, NOT fear/terror!)
            "shy" | "shyly" | "bashful" | "bashfully" | "embarrassed" | "embarrassedly"
            | "blushing" | "self-conscious" | "modest" | "demure"
            | "emotion-shy" | "emotion-embarrassed" | "emotion-shame"
            | "照れ" | "照れながら" | "恥ずかしい" | "恥ずかしそう" | "はにかむ" => {
                sliders.insert("happy".to_string(), 0.35);
                sliders.insert("surprised".to_string(), 0.25);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.2),
                    arousal: Some(-0.1),
                    cfg_scale: 1.05,
                    speed_factor: 0.92,
                })
            }

            // 5a. One-Shot Sigh / Breath
            "sigh" | "sighs" | "sighing" | "breath" | "breathe" | "breathing"
            | "exhale" | "inhale" | "sfx-sigh" | "sfx-breath"
            | "ため息" | "溜息" | "吐息" => {
                sliders.insert("sad".to_string(), 0.6);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.3),
                    arousal: Some(-0.4),
                    cfg_scale: 1.08,
                    speed_factor: 0.92,
                })
            }
            // 5b. Carry Tired / Sleepy / Sluggish / Languid
            "tired" | "tiredly" | "exhausted" | "weary" | "wearily" | "sleepy"
            | "sleepily" | "drowsy" | "sluggish" | "sluggishly" | "languid"
            | "languidly" | "resigned" | "resignedly" | "bored"
            | "眠い" | "眠そう" | "気だるげ" => {
                sliders.insert("sad".to_string(), 0.45);
                sliders.insert("happy".to_string(), 0.15);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.1),
                    arousal: Some(-0.5),
                    cfg_scale: 1.05,
                    speed_factor: 0.88,
                })
            }

            // 6a. One-Shot Cry / Sob / Weep
            "crying" | "cry" | "cries" | "weep" | "weeping" | "sob" | "sobs" | "sobbing"
            | "whimper" | "whimpers" | "whimpering" | "sfx-cry" | "sfx-crying" | "sfx-sob"
            | "泣き" | "泣" | "嗚咽" => {
                sliders.insert("sad".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.65),
                    arousal: Some(0.2),
                    cfg_scale: 1.12,
                    speed_factor: 0.90,
                })
            }
            // 6b. Carry Sad / Sorrow / Pleading
            "sad" | "sadly" | "sorrow" | "sorrowful" | "sorrowfully" | "mournful"
            | "mournfully" | "grief" | "grieving" | "heartbroken" | "melancholy"
            | "wistful" | "tearful" | "tearfully" | "ashamed" | "hurt" | "remorseful"
            | "plead" | "pleading" | "pleadingly" | "beg" | "begging" | "imploring"
            | "emotion-sad" | "emotion-sadness" | "emotion-sorrow" | "emotion-melancholy"
            | "emotion-longing" | "emotion-helplessness"
            | "悲しい" | "悲しみ" | "哀" | "懇願" => {
                sliders.insert("sad".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.5),
                    arousal: Some(-0.3),
                    cfg_scale: 1.10,
                    speed_factor: 0.90,
                })
            }
            // 6c. Carry Worried / Trembling / Timid / Hesitant
            "worried" | "worriedly" | "worry" | "concerned" | "uneasy" | "troubled"
            | "apprehensive" | "trembling" | "trembling voice" | "quavering" | "shaky"
            | "shakily" | "hesitant" | "hesitantly" | "uncertain" | "uncertainly"
            | "timid" | "timidly" | "vulnerable" | "emotion-worried"
            | "心配" | "心配そう" | "おどおど" | "震え声" => {
                sliders.insert("sad".to_string(), 0.55);
                sliders.insert("surprised".to_string(), 0.25);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.35),
                    arousal: Some(0.2),
                    cfg_scale: 1.08,
                    speed_factor: 0.94,
                })
            }

            // 7a. One-Shot Groan / Moan / Grunt
            "groan" | "groans" | "groaning" | "moan" | "moans" | "moaning"
            | "grunt" | "grunts" | "sfx-groan" | "sfx-moan"
            | "うめき" | "呻き" | "喘ぎ" => {
                sliders.insert("sad".to_string(), 0.5);
                sliders.insert("angry".to_string(), 0.25);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.35),
                    arousal: Some(0.2),
                    cfg_scale: 1.08,
                    speed_factor: 0.90,
                })
            }
            // 7b. Carry Pain / Agony / Drunk
            "strained" | "pained" | "pain" | "painful" | "painfully" | "agony"
            | "agonizing" | "agonizingly" | "distress" | "distressed" | "hoarse"
            | "hoarsely" | "weak" | "weakly" | "drunk" | "drunken" | "drunkenly"
            | "tipsy" | "slurred" | "emotion-pain"
            | "苦しげ" | "酔っ払い" => {
                sliders.insert("sad".to_string(), 0.5);
                sliders.insert("angry".to_string(), 0.25);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.45),
                    arousal: Some(0.1),
                    cfg_scale: 1.10,
                    speed_factor: 0.90,
                })
            }

            // 8. One-Shot Sniff / Sniffle
            "sniff" | "sniffs" | "sniffling" | "sniffle" | "sniffles" | "sfx-sniff"
            | "鼻をすする" => {
                sliders.insert("sad".to_string(), 0.5);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.35),
                    arousal: Some(-0.2),
                    cfg_scale: 1.08,
                    speed_factor: 0.92,
                })
            }

            // 9a. One-Shot Scream / Shriek
            "scream" | "screams" | "screaming" | "shriek" | "shrieking"
            | "sfx-scream" | "sfx-screaming" | "悲鳴" => {
                sliders.insert("angry".to_string(), 0.75);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.65),
                    arousal: Some(0.75),
                    cfg_scale: 1.15,
                    speed_factor: 1.10,
                })
            }
            // 9b. Carry Pout / Sulking / Displeased (cute sulk at normal pace, NOT fast shouting!)
            "pout" | "pouting" | "pouty" | "sulk" | "sulking" | "sulkily" | "petulant"
            | "grumpy" | "displeased" | "dissatisfied" | "dissatisfaction"
            | "dissatisfaction-hnn"
            | "不満" | "不満げ" | "拗ね" | "むすっ" => {
                sliders.insert("angry".to_string(), 0.6);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.3),
                    arousal: Some(0.3),
                    cfg_scale: 1.08,
                    speed_factor: 0.98,
                })
            }
            // 9c. Carry Angry / Furious / Stern
            "angry" | "angrily" | "anger" | "mad" | "furious" | "furiously" | "enraged"
            | "irritated" | "irritatedly" | "frustrated" | "frustratedly"
            | "impatient" | "impatiently" | "bitter" | "bitterly"
            | "hostile" | "menacing" | "menacingly" | "threatening" | "defiant" | "defiantly"
            | "stern" | "sternly" | "commanding" | "authoritative" | "gruff" | "gruffly"
            | "emotion-angry" | "emotion-anger" | "emotion-furious" | "emotion-frustrated"
            | "emotion-bitterness"
            | "怒り" | "怒" => {
                sliders.insert("angry".to_string(), 0.75);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.45),
                    arousal: Some(0.5),
                    cfg_scale: 1.12,
                    speed_factor: 1.02,
                })
            }
            // 9d. Carry Shout / Yell
            "shout" | "shouts" | "shouting" | "yell" | "yells" | "yelling" | "loud"
            | "loudly" | "style-shout" | "style-shouting" | "叫び" | "大声" => {
                sliders.insert("angry".to_string(), 0.75);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.5),
                    arousal: Some(0.65),
                    cfg_scale: 1.15,
                    speed_factor: 1.08,
                })
            }

            // 10a. One-Shot Surprise & Questioning reactions (`nina_agent` compatible)
            "surprise-wa" | "surprise-oh" | "surprise-ah" | "surprise-yo" | "exclamation" => {
                sliders.insert("surprised".to_string(), 0.65);
                sliders.insert("happy".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.25),
                    arousal: Some(0.5),
                    cfg_scale: 1.10,
                    speed_factor: 1.02,
                })
            }
            "question-oh" => {
                sliders.insert("surprised".to_string(), 0.45);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.1),
                    arousal: Some(0.2),
                    cfg_scale: 1.06,
                    speed_factor: 0.98,
                })
            }
            // 10b. Carry Surprise / Shock / Awed
            "surprise" | "surprised" | "shock" | "shocked" | "startled" | "astonished"
            | "amazed" | "astounded" | "stunned" | "dumbfounded" | "awed" | "awe"
            | "emotion-surprise" | "emotion-surprised" | "emotion-awe"
            | "驚き" | "驚" | "感嘆" | "びっくり" => {
                sliders.insert("surprised".to_string(), 0.65);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.2),
                    arousal: Some(0.5),
                    cfg_scale: 1.10,
                    speed_factor: 1.0,
                })
            }
            // 10c. Carry Questioning / Curious / Wondering
            "questioning" | "questioningly" | "questioning voice" | "wondering"
            | "wonderingly" | "curious" | "curiously" | "puzzled" | "perplexed"
            | "skeptical" | "doubtful" | "emotion-curious" | "emotion-confusion"
            | "疑問" | "不思議そう" => {
                sliders.insert("surprised".to_string(), 0.45);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.1),
                    arousal: Some(0.2),
                    cfg_scale: 1.06,
                    speed_factor: 0.98,
                })
            }

            // 11. One-Shot Gasp / Pant / Breathless
            "gasp" | "gasps" | "gasping" | "pant" | "pants" | "panting"
            | "heavy breathing" | "shortness of breath" | "breathless" | "breathlessly"
            | "out of breath" | "sharp intake of breath" | "sfx-gasp" | "sfx-pant"
            | "ハッ" | "息をのむ" | "息切れ" => {
                sliders.insert("surprised".to_string(), 0.55);
                sliders.insert("fear".to_string(), 0.25);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.1),
                    arousal: Some(0.55),
                    cfg_scale: 1.10,
                    speed_factor: 1.0,
                })
            }

            // 12a. One-Shot Stutter / Stammer
            "stutter" | "stutters" | "stuttering" | "stammer" | "stammering"
            | "どもり" => {
                sliders.insert("fear".to_string(), 0.55);
                sliders.insert("surprised".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.35),
                    arousal: Some(0.4),
                    cfg_scale: 1.10,
                    speed_factor: 0.98,
                })
            }
            // 12b. Carry Fear / Scared / Panicked / Nervous
            "fear" | "fearful" | "scared" | "terrified" | "frightened" | "afraid"
            | "horrified" | "panicked" | "panic" | "panicking" | "agitated" | "agitatedly"
            | "nervous" | "nervously" | "anxious" | "anxiously" | "flustered"
            | "urgent" | "urgently" | "desperate" | "desperately"
            | "emotion-fear" | "emotion-anxious" | "emotion-nervous" | "emotion-panicked"
            | "恐れ" | "怖" | "恐怖" | "慌て" | "動揺" => {
                sliders.insert("fear".to_string(), 0.65);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.5),
                    arousal: Some(0.5),
                    cfg_scale: 1.12,
                    speed_factor: 1.02,
                })
            }

            // 13. Carry Disgust
            "disgust" | "disgusted" | "emotion-disgust" | "嫌悪" => {
                sliders.insert("disgust".to_string(), 0.65);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.6),
                    arousal: Some(0.2),
                    cfg_scale: 1.10,
                    speed_factor: 1.0,
                })
            }

            // 14a. One-Shot Scoff / Eye roll
            "scoff" | "scoffs" | "rolling eyes" | "eye roll" => {
                sliders.insert("angry".to_string(), 0.3);
                sliders.insert("sad".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.3),
                    arousal: Some(0.1),
                    cfg_scale: 1.08,
                    speed_factor: 0.96,
                })
            }
            // 14b. Carry Sarcastic / Mocking / Exasperated / Cold
            "sarcastic" | "sarcastically" | "sarcasm" | "ironic" | "mocking" | "mockingly"
            | "smug" | "smugly" | "smirk" | "exasperated" | "exasperatedly"
            | "annoyed" | "cold" | "coldly" | "grim" | "grimly" | "dismissive" | "dismissively"
            | "suspicious" | "suspiciously" | "style-sarcastic" | "style-mocking"
            | "emotion-annoyed" | "皮肉" | "呆れ" => {
                sliders.insert("angry".to_string(), 0.35);
                sliders.insert("sad".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.3),
                    arousal: Some(-0.05),
                    cfg_scale: 1.08,
                    speed_factor: 0.95,
                })
            }

            // 15a. Carry Dramatic / Solemn / Singing / Humming
            "dramatic" | "dramatic tone" | "solemn" | "solemnly" | "echo" | "reverb" => {
                sliders.insert("surprised".to_string(), 0.3);
                sliders.insert("sad".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.0),
                    arousal: Some(0.2),
                    cfg_scale: 1.12,
                    speed_factor: 0.90,
                })
            }
            "advertisement" | "commercial" | "sings" | "singing" | "hum" | "humming"
            | "style-singing" | "style-hum" | "sfx-humming" | "鼻歌" => {
                sliders.insert("happy".to_string(), 0.45);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.45),
                    arousal: Some(0.25),
                    cfg_scale: 1.08,
                    speed_factor: 1.02,
                })
            }
            // 15b. Carry Relieved / Satisfied / Peaceful
            "relieved" | "relief" | "satisfied" | "content" | "contented" | "contentedly"
            | "peaceful" | "serene" | "emotion-relieved" | "emotion-relief"
            | "安堵" | "ほっとした" | "満足" => {
                sliders.insert("happy".to_string(), 0.4);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.35),
                    arousal: Some(-0.25),
                    cfg_scale: 1.05,
                    speed_factor: 0.94,
                })
            }

            // 16. Reset / Calm / Neutral
            "reset" | "narration" | "narrator" | "calm" | "calmly" | "neutral" | "normal"
            | "serious" | "seriously" | "thoughtful" | "thoughtfully" | "determined"
            | "deadpan" | "monotone" | "confused" | "emotion-calm" | "emotion-contemplation"
            | "emotion-determination" | "style-narration" | "落ち着き" => Some(EmotionProfile {
                layer: TagLayer::Reset,
                sliders,
                valence: None,
                arousal: None,
                cfg_scale: 1.0,
                speed_factor: 1.0,
            }),

            // 17. Carry Pacing Modifiers
            "slowly" | "slow" | "deliberate" | "deliberately" | "prosody-slow"
            | "prosody-speed_slow" | "prosody-speed_very_slow" | "ゆっくり" => {
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: None,
                    arousal: None,
                    cfg_scale: 1.0,
                    speed_factor: 0.75,
                })
            }
            "quickly" | "quick" | "fast" | "rapid-fire" | "rushed" | "hurried"
            | "hurriedly" | "prosody-fast" | "prosody-speed_fast" | "prosody-speed_very_fast"
            | "早口" => Some(EmotionProfile {
                layer: TagLayer::Carry,
                sliders,
                valence: None,
                arousal: None,
                cfg_scale: 1.0,
                speed_factor: 1.20,
            }),

            // 18a. One-Shot Confirmation / Backchannel (`[confirmation-en]`, `[nod]`)
            "confirmation-en" | "confirmation" => {
                sliders.insert("happy".to_string(), 0.25);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.2),
                    arousal: Some(-0.1),
                    cfg_scale: 1.04,
                    speed_factor: 0.98,
                })
            }

            // 18b. Pure sound/action tags (stripped from text without altering emotion)
            "clear throat" | "clears throat" | "clearing throat" | "cough" | "coughs"
            | "coughing" | "pause" | "pauses" | "hesitates" | "beat" | "silence"
            | "sneeze" | "sneezes" | "sneezing" | "yawn" | "yawns" | "yawning"
            | "gulp" | "gulps" | "swallow" | "swallows" | "tsk" | "tut"
            | "click tongue" | "tongue click" | "lip smack" | "lip noise"
            | "kiss" | "smooch" | "lick" | "licking" | "chew" | "chewing" | "wet sound"
            | "nod" | "nods" | "nodding" | "agree" | "agreement" | "backchannel"
            | "mm-hmm" | "uh-huh" | "phone" | "telephone" | "sfx-cough" | "sfx-sneeze"
            | "sfx-burping" => Some(EmotionProfile {
                layer: TagLayer::Neutral,
                sliders,
                valence: None,
                arousal: None,
                cfg_scale: 1.0,
                speed_factor: 1.0,
            }),

            _ => None,
        }
    }

    /// Extracts raw tags and cleans tags/orphan brackets from a single segment.
    fn extract_tags_and_clean(segment: &str) -> (String, Vec<String>) {
        let mut detected_tags = Vec::new();
        let mut working_text = SE_WRAPPER_REGEX.replace_all(segment, " ").to_string();

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
        working_text = MISSING_OPEN_BRACKET_REGEX
            .replace_all(&working_text, "$prefix ")
            .to_string();

        // 3. Rescue and strip tags with missing closing bracket (e.g. "[whisper Hello")
        for cap in MISSING_CLOSE_BRACKET_REGEX.captures_iter(&working_text) {
            if let Some(m) = cap.name("itag") {
                let raw_tag = m.as_str().trim().to_lowercase();
                if !raw_tag.is_empty() {
                    detected_tags.push(raw_tag);
                }
            }
        }
        working_text = MISSING_CLOSE_BRACKET_REGEX
            .replace_all(&working_text, " $suffix")
            .to_string();

        // 4. Sanitize whitespace and orphan boundary bracket remnants
        let cleaned_collapsed = MULTI_SPACE_REGEX.replace_all(&working_text, " ").to_string();
        let trimmed = cleaned_collapsed.trim();
        let sanitized = ORPHAN_BRACKETS_REGEX.replace_all(trimmed, "").to_string();
        let cleaned_trimmed = sanitized.trim().to_string();

        (cleaned_trimmed, detected_tags)
    }

    fn build_parsed_prompt(
        cleaned_text: String,
        detected_tags: Vec<String>,
        effective_tags: Vec<String>,
        final_profile: Option<EmotionProfile>,
    ) -> ParsedPrompt {
        if let Some(profile) = final_profile {
            let emotion_enabled = !profile.sliders.is_empty();
            ParsedPrompt {
                cleaned_text,
                emotion_enabled,
                emotion_sliders: profile.sliders,
                emotion_valence: if emotion_enabled { profile.valence } else { None },
                emotion_arousal: if emotion_enabled { profile.arousal } else { None },
                emotion_cfg_scale: profile.cfg_scale,
                speed_factor: profile.speed_factor,
                accurate_mode: false,
                detected_tags,
                effective_tags,
            }
        } else {
            ParsedPrompt {
                cleaned_text,
                emotion_enabled: false,
                emotion_sliders: HashMap::new(),
                emotion_valence: None,
                emotion_arousal: None,
                emotion_cfg_scale: 1.0,
                speed_factor: 1.0,
                accurate_mode: false,
                detected_tags,
                effective_tags,
            }
        }
    }

    /// Parses a single utterance chunk into `ParsedPrompt` with 3-layer blending:
    /// Layer 1 (`[mood:<tag>]`) + Layer 2 (Carry Tag) + Layer 3 (One-Shot Tag).
    pub fn parse(input: &str) -> ParsedPrompt {
        let (cleaned_trimmed, detected_tags) = Self::extract_tags_and_clean(input);

        let mut base_mood: Option<(String, EmotionProfile)> = None;
        let mut active_carries: Vec<(String, EmotionProfile)> = Vec::new();
        let mut active_oneshots: Vec<(String, EmotionProfile)> = Vec::new();
        let mut reset_triggered = false;

        for tag in &detected_tags {
            if let Some(mood_inner) = tag.strip_prefix("mood:") {
                let base_tag = Self::strip_modifier_prefix(mood_inner.trim());
                if let Some(profile) = Self::classify_tag(base_tag) {
                    if matches!(profile.layer, TagLayer::Carry | TagLayer::OneShot) {
                        base_mood = Some((base_tag.to_string(), Self::scale_base_mood(&profile)));
                    }
                }
                continue;
            }

            let base_tag = Self::strip_modifier_prefix(tag.as_str());
            if let Some(profile) = Self::classify_tag(base_tag) {
                match profile.layer {
                    TagLayer::Carry => {
                        reset_triggered = false;
                        active_carries.push((base_tag.to_string(), profile));
                    }
                    TagLayer::OneShot => {
                        active_oneshots.push((base_tag.to_string(), profile));
                    }
                    TagLayer::Reset => {
                        reset_triggered = true;
                        active_carries.clear();
                        active_oneshots.clear();
                    }
                    TagLayer::Neutral => {}
                }
            }
        }

        let eff_mood = if reset_triggered {
            None
        } else {
            base_mood.as_ref()
        };

        let carry_merged: Option<(String, EmotionProfile)> = match active_carries.as_slice() {
            [] => None,
            [(name, single)] => Some((name.clone(), single.clone())),
            [( _, first), .., (last_name, last)] => Some((
                last_name.clone(),
                Self::blend_profiles(
                    first,
                    Self::BASE_CARRY_WEIGHT,
                    last,
                    Self::OVERRIDE_CARRY_WEIGHT,
                ),
            )),
        };

        let oneshot_merged = active_oneshots.last().cloned();

        let mut effective_tags = Vec::new();
        if carry_merged.is_none() {
            if let Some((m_name, _)) = eff_mood {
                effective_tags.push(m_name.clone());
            }
        }
        if let Some((c_name, _)) = &carry_merged {
            effective_tags.push(c_name.clone());
        }
        if let Some((o_name, _)) = &oneshot_merged {
            effective_tags.push(o_name.clone());
        }

        let final_profile = Self::blend_three_layers(
            eff_mood.map(|(_, p)| p),
            carry_merged.as_ref().map(|(_, p)| p),
            oneshot_merged.as_ref().map(|(_, p)| p),
        );

        Self::build_parsed_prompt(cleaned_trimmed, detected_tags, effective_tags, final_profile)
    }

    fn blend_three_layers(
        mood: Option<&EmotionProfile>,
        carry: Option<&EmotionProfile>,
        oneshot: Option<&EmotionProfile>,
    ) -> Option<EmotionProfile> {
        match (mood, carry, oneshot) {
            (Some(m), Some(c), Some(o)) => {
                Some(Self::blend_weighted_profiles(&[(m, 0.20), (c, 0.30), (o, 0.50)]))
            }
            (Some(m), Some(c), None) => {
                Some(Self::blend_weighted_profiles(&[(m, 0.25), (c, 0.75)]))
            }
            (Some(m), None, Some(o)) => {
                Some(Self::blend_weighted_profiles(&[(m, 0.35), (o, 0.65)]))
            }
            (None, Some(c), Some(o)) => Some(Self::blend_profiles(
                c,
                Self::CARRY_BLEND_WEIGHT,
                o,
                Self::ONESHOT_BLEND_WEIGHT,
            )),
            (Some(m), None, None) => Some(m.clone()),
            (None, Some(c), None) => Some(c.clone()),
            (None, None, Some(o)) => Some(o.clone()),
            (None, None, None) => None,
        }
    }

    /// Splits raw input on sentence terminators (`。`, `！`, `？`, `\n`, `!`, `?`, and standalone `.`)
    /// outside enclosed tags while preserving decimals (`10.5`), abbreviations (`K.I.T.T.`), and ellipses (`...`).
    pub fn split_raw_sentences(input: &str) -> Vec<String> {
        let chars: Vec<(usize, char)> = input.char_indices().collect();
        let n = chars.len();
        let mut segments = Vec::new();
        let mut start_byte = 0;
        let mut i = 0;
        let mut bracket_depth: usize = 0;
        let mut in_asterisk = false;

        while i < n {
            let (_, ch) = chars[i];
            match ch {
                '[' | '【' | '(' | '（' => {
                    bracket_depth = bracket_depth.saturating_add(1);
                    i += 1;
                    continue;
                }
                ']' | '】' | ')' | '）' => {
                    bracket_depth = bracket_depth.saturating_sub(1);
                    i += 1;
                    continue;
                }
                '*' => {
                    in_asterisk = !in_asterisk;
                    i += 1;
                    continue;
                }
                _ => {}
            }

            if bracket_depth == 0 && !in_asterisk {
                if matches!(ch, '。' | '！' | '？' | '!' | '?' | '\n' | '\r') {
                    let mut end_i = i + 1;
                    while end_i < n
                        && matches!(
                            chars[end_i].1,
                            '。' | '！' | '？' | '!' | '?' | '\n' | '\r'
                        )
                    {
                        end_i += 1;
                    }
                    let end_byte = if end_i < n {
                        chars[end_i].0
                    } else {
                        input.len()
                    };
                    let seg = input[start_byte..end_byte].trim();
                    if !seg.is_empty() {
                        segments.push(seg.to_string());
                    }
                    start_byte = end_byte;
                    i = end_i;
                    continue;
                } else if ch == '.' {
                    let prev_is_dot = i > 0 && chars[i - 1].1 == '.';
                    let next_is_dot = i + 1 < n && chars[i + 1].1 == '.';
                    let next_is_ws = i + 1 < n && chars[i + 1].1.is_whitespace();
                    // Avoid splitting single-letter abbreviations like "U.S. " or "Mr." if preceded by dot pattern
                    let prev_abbrev = i >= 2 && chars[i - 2].1 == '.';

                    if !prev_is_dot && !next_is_dot && !prev_abbrev && next_is_ws {
                        let end_byte = chars[i + 1].0;
                        let seg = input[start_byte..end_byte].trim();
                        if !seg.is_empty() {
                            segments.push(seg.to_string());
                        }
                        start_byte = end_byte;
                        i += 1;
                        continue;
                    }
                }
            }

            i += 1;
        }

        if start_byte < input.len() {
            let tail = input[start_byte..].trim();
            if !tail.is_empty() {
                segments.push(tail.to_string());
            }
        }

        segments
    }

    /// Further splits a sentence segment right before any mid-sentence `Carry` or `Reset` tag
    /// when speakable text has already appeared earlier in the segment
    /// (e.g. `"[whisper] 秘密だよ… [calm] さて本題だよ。"` -> 2 chunks).
    fn split_on_mid_sentence_carry_shifts(segment: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut last_split = 0;

        for cap in FULL_TAG_REGEX.captures_iter(segment) {
            let Some(m) = cap.get(0) else { continue };
            let raw_tag = cap
                .name("tag")
                .or_else(|| cap.name("ztag"))
                .or_else(|| cap.name("ptag"))
                .or_else(|| cap.name("fptag"))
                .or_else(|| cap.name("atag"))
                .map(|t| t.as_str().trim().to_lowercase())
                .unwrap_or_default();

            if raw_tag.is_empty() || raw_tag.starts_with("mood:") {
                continue;
            }

            let base_tag = Self::strip_modifier_prefix(&raw_tag);
            let is_carry_or_reset = Self::classify_tag(base_tag)
                .map(|p| matches!(p.layer, TagLayer::Carry | TagLayer::Reset))
                .unwrap_or(false);

            if is_carry_or_reset {
                let preceding = &segment[last_split..m.start()];
                let (cleaned_preceding, _) = Self::extract_tags_and_clean(preceding);
                if SPEAKABLE_CHAR_REGEX.is_match(&cleaned_preceding) {
                    out.push(preceding.trim().to_string());
                    last_split = m.start();
                }
            }
        }

        let remainder = segment[last_split..].trim();
        if !remainder.is_empty() {
            out.push(remainder.to_string());
        }

        if out.is_empty() && !segment.trim().is_empty() {
            out.push(segment.trim().to_string());
        }

        out
    }

    /// Parses a single- or multi-sentence utterance into sentence-level `ParsedPrompt` chunks,
    /// propagating Carry Tags and `[mood:<tag>]` across sentences while keeping One-Shot Tags
    /// strictly local to the sentence in which they appear and honoring `[calm]` / `[reset]`.
    pub fn parse_chunks(input: &str) -> Vec<ParsedPrompt> {
        let without_se = SE_WRAPPER_REGEX.replace_all(input, " ").to_string();
        let raw_sentences = Self::split_raw_sentences(&without_se);
        if raw_sentences.is_empty() {
            return vec![Self::parse(input)];
        }

        let mut fine_segments: Vec<String> = Vec::new();
        for sent in raw_sentences {
            fine_segments.extend(Self::split_on_mid_sentence_carry_shifts(&sent));
        }

        // Merge tag-only segments (no speakable characters):
        // - Leading or Carry/Reset/Mood-bearing tag-only segments attach forward to the next speakable segment.
        // - Trailing tag-only segments (e.g. "...they'll hear us. [sigh]") attach backward to the previous speakable segment.
        let mut merged_segments: Vec<String> = Vec::new();
        let mut pending_prefix = String::new();

        for seg in fine_segments {
            let (cleaned, tags) = Self::extract_tags_and_clean(&seg);
            if SPEAKABLE_CHAR_REGEX.is_match(&cleaned) {
                if pending_prefix.is_empty() {
                    merged_segments.push(seg);
                } else {
                    merged_segments.push(format!("{pending_prefix} {seg}"));
                    pending_prefix.clear();
                }
            } else {
                // Tag-only or punctuation-only segment
                let has_forward_steering = tags.iter().any(|t| {
                    if t.starts_with("mood:") {
                        return true;
                    }
                    let base = Self::strip_modifier_prefix(t);
                    Self::classify_tag(base)
                        .map(|p| matches!(p.layer, TagLayer::Carry | TagLayer::Reset))
                        .unwrap_or(false)
                });

                if merged_segments.is_empty() || has_forward_steering {
                    if pending_prefix.is_empty() {
                        pending_prefix = seg;
                    } else {
                        pending_prefix.push(' ');
                        pending_prefix.push_str(&seg);
                    }
                } else if let Some(last) = merged_segments.last_mut() {
                    last.push(' ');
                    last.push_str(&seg);
                }
            }
        }

        if !pending_prefix.is_empty() {
            if let Some(last) = merged_segments.last_mut() {
                last.push(' ');
                last.push_str(&pending_prefix);
            } else {
                return vec![Self::parse(&without_se)];
            }
        }

        if merged_segments.is_empty() {
            return vec![Self::parse(&without_se)];
        }

        // Stateful 3-layer propagation across merged speakable segments
        let mut persistent_mood: Option<(String, EmotionProfile)> = None;
        let mut persistent_carry: Option<(String, EmotionProfile)> = None;
        let mut calm_reset_active = false;
        let mut results = Vec::with_capacity(merged_segments.len());

        for seg in merged_segments {
            let (cleaned_text, detected_tags) = Self::extract_tags_and_clean(&seg);
            let mut local_carries: Vec<(String, EmotionProfile)> = Vec::new();
            let mut local_oneshots: Vec<(String, EmotionProfile)> = Vec::new();

            for tag in &detected_tags {
                if let Some(mood_inner) = tag.strip_prefix("mood:") {
                    let base_tag = Self::strip_modifier_prefix(mood_inner.trim());
                    if let Some(profile) = Self::classify_tag(base_tag) {
                        if matches!(profile.layer, TagLayer::Carry | TagLayer::OneShot) {
                            persistent_mood =
                                Some((base_tag.to_string(), Self::scale_base_mood(&profile)));
                        }
                    }
                    continue;
                }

                let base_tag = Self::strip_modifier_prefix(tag.as_str());
                if let Some(profile) = Self::classify_tag(base_tag) {
                    match profile.layer {
                        TagLayer::Carry => {
                            calm_reset_active = false;
                            local_carries.push((base_tag.to_string(), profile));
                        }
                        TagLayer::OneShot => {
                            local_oneshots.push((base_tag.to_string(), profile));
                        }
                        TagLayer::Reset => {
                            calm_reset_active = true;
                            persistent_carry = None;
                            local_carries.clear();
                            local_oneshots.clear();
                        }
                        TagLayer::Neutral => {}
                    }
                }
            }

            match local_carries.as_slice() {
                [] => {}
                [(name, single)] => {
                    persistent_carry = Some((name.clone(), single.clone()));
                }
                [(_, first), .., (last_name, last)] => {
                    persistent_carry = Some((
                        last_name.clone(),
                        Self::blend_profiles(
                            first,
                            Self::BASE_CARRY_WEIGHT,
                            last,
                            Self::OVERRIDE_CARRY_WEIGHT,
                        ),
                    ));
                }
            }

            let eff_mood = if calm_reset_active {
                None
            } else {
                persistent_mood.as_ref()
            };
            let eff_carry = persistent_carry.as_ref();
            let eff_oneshot = local_oneshots.last();

            let mut effective_tags = Vec::new();
            if eff_carry.is_none() {
                if let Some((m_name, _)) = eff_mood {
                    effective_tags.push(m_name.clone());
                }
            }
            if let Some((c_name, _)) = eff_carry {
                effective_tags.push(c_name.clone());
            }
            if let Some((o_name, _)) = eff_oneshot {
                effective_tags.push(o_name.clone());
            }

            let final_profile = Self::blend_three_layers(
                eff_mood.map(|(_, p)| p),
                eff_carry.map(|(_, p)| p),
                eff_oneshot.map(|(_, p)| p),
            );

            results.push(Self::build_parsed_prompt(
                cleaned_text,
                detected_tags,
                effective_tags,
                final_profile,
            ));
        }

        results
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
        assert!(parsed.emotion_enabled);
        assert!(!parsed.accurate_mode);
        assert_eq!(parsed.speed_factor, 0.88);
        assert_eq!(parsed.emotion_cfg_scale, 1.0);
        assert_eq!(parsed.emotion_sliders.get("happy"), Some(&0.2));
    }

    #[test]
    fn test_parse_laughter() {
        let input = "[laughter] えへへ、パパ大好き！";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "えへへ、パパ大好き！");
        assert_eq!(parsed.detected_tags, vec!["laughter"]);
        assert!(parsed.emotion_enabled);
        assert_eq!(parsed.emotion_cfg_scale, 1.12);
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
        assert!(!parsed.emotion_enabled);
        // Must remain false even on plain text to keep Zonos 2 speaker timbre consistent
        assert!(!parsed.accurate_mode);
        assert_eq!(parsed.speed_factor, 1.0);
        assert_eq!(parsed.emotion_cfg_scale, 1.0);
        assert!(parsed.emotion_sliders.is_empty());
    }

    #[test]
    fn test_parse_skyrimnet_chatterbox_tags() {
        // Multi-word tag with space, sound tags in middle and end of sentence
        let input = "[whispering] Keep your voice down - [shush] - they’ll hear us. [sigh]";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "Keep your voice down - - they’ll hear us.");
        assert_eq!(parsed.detected_tags, vec!["whispering", "shush", "sigh"]);
        // Carry [whispering] (happy: 0.2) + OneShot [sigh] (sad: 0.6) are blended (0.35 / 0.65)
        assert_eq!(parsed.emotion_sliders.get("happy"), Some(&0.07));
        assert_eq!(parsed.emotion_sliders.get("sad"), Some(&0.39));

        let input2 = "[happy] [clear throat] So… we agree? [chuckle] Great.";
        let parsed2 = EmotionParser::parse(input2);
        assert_eq!(parsed2.cleaned_text, "So… we agree? Great.");
        assert_eq!(parsed2.detected_tags, vec!["happy", "clear throat", "chuckle"]);
        // Carry [happy] (0.7 * 0.35) + OneShot [chuckle] (0.5 * 0.65) = 0.57
        assert_eq!(parsed2.emotion_sliders.get("happy"), Some(&0.57));
    }

    #[test]
    fn test_carry_and_oneshot_blending_whisper_laughter() {
        let input = "[whisper] [laughter] ふふ、くすぐったいよ";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "ふふ、くすぐったいよ");
        assert_eq!(parsed.detected_tags, vec!["whisper", "laughter"]);
        assert!(parsed.emotion_enabled);
        // happy: 0.2 * 0.35 + 0.7 * 0.65 = 0.525
        assert_eq!(parsed.emotion_sliders.get("happy"), Some(&0.525));
        // speed_factor: 0.88 * 0.35 + 1.0 * 0.65 = 0.958
        assert_eq!(parsed.speed_factor, 0.958);
        // cfg_scale: 1.0 * 0.35 + 1.12 * 0.65 = 1.078
        assert_eq!(parsed.emotion_cfg_scale, 1.078);
    }

    #[test]
    fn test_reset_tag_clears_previous_carry() {
        let input = "[whisper] 秘密だよ… [calm] さて、普通の話に戻ろうか。";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "秘密だよ… さて、普通の話に戻ろうか。");
        assert!(!parsed.emotion_enabled);
        assert!(parsed.emotion_sliders.is_empty());
        assert_eq!(parsed.speed_factor, 1.0);
        assert!(!parsed.accurate_mode);

        // Multi-chunk parser splits before mid-sentence [calm] so the first half stays whispered!
        let chunks = EmotionParser::parse_chunks(input);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].cleaned_text, "秘密だよ…");
        assert!(chunks[0].emotion_enabled);
        assert_eq!(chunks[0].speed_factor, 0.88);
        assert_eq!(chunks[1].cleaned_text, "さて、普通の話に戻ろうか。");
        assert!(!chunks[1].emotion_enabled);
        assert_eq!(chunks[1].speed_factor, 1.0);
    }

    #[test]
    fn test_parse_chunks_propagates_carry_isolates_oneshot_and_blends_base_mood() {
        let input = "[mood:gently] [whisper] ねえパパ。 [laughter] ふふ、くすぐったいよ！ ずっとこうしてたいな。 [calm] さて、ご飯にしようか。 何が食べたい？";
        let chunks = EmotionParser::parse_chunks(input);
        assert_eq!(chunks.len(), 5);

        // Chunk 0: Base gently (0.25) + Carry whisper (0.75)
        assert_eq!(chunks[0].cleaned_text, "ねえパパ。");
        assert_eq!(chunks[0].effective_tags, vec!["whisper"]);
        assert!(chunks[0].emotion_enabled);
        let c0_happy = *chunks[0].emotion_sliders.get("happy").unwrap();
        assert!(c0_happy > 0.2 && c0_happy < 0.35);

        // Chunk 1: Base gently (0.20) + Carry whisper (0.30) + OneShot laughter (0.50)
        assert_eq!(chunks[1].cleaned_text, "ふふ、くすぐったいよ！");
        assert_eq!(chunks[1].effective_tags, vec!["whisper", "laughter"]);
        let c1_happy = *chunks[1].emotion_sliders.get("happy").unwrap();
        assert!(c1_happy > c0_happy, "Laughter OneShot boosts happy on chunk 1 only");

        // Chunk 2: Carry whisper returns cleanly without laughter bleeding into chunk 2!
        assert_eq!(chunks[2].cleaned_text, "ずっとこうしてたいな。");
        assert_eq!(chunks[2].effective_tags, vec!["whisper"]);
        assert_eq!(chunks[2].emotion_sliders.get("happy"), Some(&c0_happy));

        // Chunk 3 & 4: [calm] resets both Carry and BaseMood until a new Carry appears
        assert_eq!(chunks[3].cleaned_text, "さて、ご飯にしようか。");
        assert!(!chunks[3].emotion_enabled);
        assert!(chunks[3].effective_tags.is_empty());

        assert_eq!(chunks[4].cleaned_text, "何が食べたい？");
        assert!(!chunks[4].emotion_enabled);
        assert!(!chunks[4].accurate_mode);
    }

    #[test]
    fn test_nina_tuned_shy_pout_gently_and_hyphen_aliases() {
        // [shy] must NOT map to fear/terror
        let shy = EmotionParser::parse("[shy] そ、そんなに見つめられたら恥ずかしいよ…");
        assert!(shy.emotion_sliders.get("fear").is_none());
        assert_eq!(shy.emotion_sliders.get("happy"), Some(&0.35));
        assert_eq!(shy.emotion_sliders.get("surprised"), Some(&0.25));
        assert_eq!(shy.emotion_cfg_scale, 1.05);

        // [pout] / [dissatisfaction-hnn] must NOT speed up to 1.15x shouting
        let pout = EmotionParser::parse("[dissatisfaction-hnn] むー、いじわる！");
        assert_eq!(pout.emotion_sliders.get("angry"), Some(&0.6));
        assert_eq!(pout.speed_factor, 0.98);
        assert_eq!(pout.emotion_cfg_scale, 1.08);

        // [surprise-wa], [question-oh], [confirmation-en] are all recognized
        let surp = EmotionParser::parse("[surprise-wa] ええっ、本当！？");
        assert!(surp.emotion_enabled);
        assert!(surp.emotion_sliders.contains_key("surprised"));

        let ques = EmotionParser::parse("[question-oh] あれ、どこ行くの？");
        assert!(ques.emotion_enabled);

        let conf = EmotionParser::parse("[confirmation-en] うん、わかったよ。");
        assert!(conf.emotion_enabled);
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
        assert_eq!(parsed.emotion_cfg_scale, 1.12);
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
