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
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TagLayer {
    /// Persistent tone / emotion / style that carries across the utterance
    Carry,
    /// Instantaneous vocal reaction applied on top of Carry
    OneShot,
    /// Resets any active Carry emotion back to calm/neutral
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
            accurate_mode: true,
            detected_tags: Vec::new(),
        }
    }
}

pub struct EmotionParser;

impl EmotionParser {
    const CARRY_BLEND_WEIGHT: f32 = 0.35;
    const ONESHOT_BLEND_WEIGHT: f32 = 0.65;
    const BASE_CARRY_WEIGHT: f32 = 0.30;
    const OVERRIDE_CARRY_WEIGHT: f32 = 0.70;

    fn strip_modifier_prefix(tag: &str) -> &str {
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

    fn blend_opt(a: Option<f32>, wa: f32, b: Option<f32>, wb: f32) -> Option<f32> {
        match (a, b) {
            (Some(va), Some(vb)) => Some(Self::round3((va * wa + vb * wb).clamp(-1.0, 1.0))),
            (Some(va), None) => Some(Self::round3(va.clamp(-1.0, 1.0))),
            (None, Some(vb)) => Some(Self::round3(vb.clamp(-1.0, 1.0))),
            (None, None) => None,
        }
    }

    fn blend_profiles(p1: &EmotionProfile, w1: f32, p2: &EmotionProfile, w2: f32) -> EmotionProfile {
        let mut merged = HashMap::new();
        for (k, v1) in &p1.sliders {
            let v2 = p2.sliders.get(k).copied().unwrap_or(0.0);
            let val = v1 * w1 + v2 * w2;
            if val > 0.001 {
                merged.insert(k.clone(), val);
            }
        }
        for (k, v2) in &p2.sliders {
            if !merged.contains_key(k) {
                let val = v2 * w2;
                if val > 0.001 {
                    merged.insert(k.clone(), val);
                }
            }
        }

        let sliders = Self::normalize_sliders(merged);

        EmotionProfile {
            layer: p2.layer,
            sliders,
            valence: Self::blend_opt(p1.valence, w1, p2.valence, w2),
            arousal: Self::blend_opt(p1.arousal, w1, p2.arousal, w2),
            cfg_scale: Self::round3(p1.cfg_scale * w1 + p2.cfg_scale * w2),
            speed_factor: Self::round3(p1.speed_factor * w1 + p2.speed_factor * w2),
        }
    }

    /// Maps a normalized base tag to its `EmotionProfile` (Carry, OneShot, Reset, or Neutral).
    pub fn classify_tag(base_tag: &str) -> Option<EmotionProfile> {
        let mut sliders = HashMap::new();
        match base_tag {
            // 1a. One-Shot Laugh
            "laughter" | "laugh" | "laughs" | "laughing" | "sfx-laughter" | "sfx-laugh"
            | "笑" | "笑い" => {
                sliders.insert("happy".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.7),
                    arousal: Some(0.5),
                    cfg_scale: 1.2,
                    speed_factor: 1.0,
                })
            }
            // 1b. Carry Happy / Joy / Proud / Confident / Glad
            "happy" | "happily" | "joy" | "joyful" | "joyfully" | "cheer" | "cheerful"
            | "cheerfully" | "glad" | "gladly" | "delighted" | "delightedly" | "pleased"
            | "smile" | "smiling" | "hopeful" | "hopefully" | "proud" | "proudly"
            | "confident" | "confidently" | "emotion-happy" | "emotion-joy"
            | "emotion-cheerful" | "emotion-proud"
            | "喜" | "喜び" | "歓喜" | "楽しげ" | "嬉しそう" => {
                sliders.insert("happy".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.6),
                    arousal: Some(0.3),
                    cfg_scale: 1.2,
                    speed_factor: 1.0,
                })
            }

            // 2a. One-Shot Chuckle / Giggle / Snicker
            "chuckle" | "chuckles" | "chuckling" | "giggle" | "giggles" | "giggling"
            | "snicker" | "snickers" | "sfx-chuckle" | "sfx-giggle" | "style-chuckle"
            | "style-giggle" | "くすくす" | "含み笑い" => {
                sliders.insert("happy".to_string(), 0.5);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.5),
                    arousal: Some(0.2),
                    cfg_scale: 1.15,
                    speed_factor: 1.0,
                })
            }
            // 2b. Carry Playful / Teasing / Coaxing / Flirty
            "playful" | "playfully" | "teasing" | "teasingly" | "tease" | "coaxing"
            | "coaxingly" | "coax" | "sweetly" | "coy" | "coyly" | "flirtatious"
            | "flirtatiously" | "flirty" | "amused" | "amusedly" | "style-playful"
            | "style-teasing" | "からかい" | "甘え" => {
                sliders.insert("happy".to_string(), 0.5);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.5),
                    arousal: Some(0.2),
                    cfg_scale: 1.15,
                    speed_factor: 1.0,
                })
            }

            // 3. Carry Excited / Enthusiastic
            "excited" | "excitedly" | "gleeful" | "gleefully" | "ecstatic"
            | "enthusiastic" | "enthusiastically" | "emotion-excited"
            | "興奮" | "わくわく" => {
                sliders.insert("happy".to_string(), 0.8);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.8),
                    arousal: Some(0.7),
                    cfg_scale: 1.2,
                    speed_factor: 1.10,
                })
            }

            // 4a. One-Shot Shush
            "shush" | "shh" => {
                sliders.insert("happy".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(0.1),
                    arousal: Some(-0.5),
                    cfg_scale: 1.1,
                    speed_factor: 0.88,
                })
            }
            // 4b. Carry Whisper / Soft / Quiet / Gentle / Tender / Sultry
            "whisper" | "whispering" | "whispers" | "whispered" | "hushed"
            | "gentle" | "gently" | "sweet" | "soft" | "softly" | "tender" | "tenderly"
            | "warm" | "warmly" | "affectionate" | "affectionately" | "loving" | "lovingly"
            | "seductive" | "seductively" | "sultry" | "reverent" | "reverently"
            | "quiet" | "quietly" | "murmur" | "murmurs" | "murmuring"
            | "mutter" | "mutters" | "muttering" | "mumble" | "mumbles" | "mumbling"
            | "conspiratorial" | "conspiratorially" | "under breath" | "muffled"
            | "style-whisper"
            | "囁き" | "ささやき" | "優しく" | "静かに" => {
                sliders.insert("happy".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.3),
                    arousal: Some(-0.5),
                    cfg_scale: 1.1,
                    speed_factor: 0.88,
                })
            }

            // 5a. One-Shot Sigh / Breath
            "sigh" | "sighs" | "sighing" | "breath" | "breathe" | "breathing"
            | "exhale" | "inhale" | "sfx-sigh" | "sfx-breath"
            | "ため息" | "吐息" => {
                sliders.insert("sad".to_string(), 0.6);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.3),
                    arousal: Some(-0.5),
                    cfg_scale: 1.15,
                    speed_factor: 0.90,
                })
            }
            // 5b. Carry Tired / Sleepy / Sluggish / Languid
            "tired" | "tiredly" | "exhausted" | "weary" | "wearily" | "sleepy"
            | "sleepily" | "drowsy" | "sluggish" | "sluggishly" | "languid"
            | "languidly" | "resigned" | "resignedly" | "bored"
            | "眠そう" | "気だるげ" => {
                sliders.insert("sad".to_string(), 0.6);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.2),
                    arousal: Some(-0.6),
                    cfg_scale: 1.15,
                    speed_factor: 0.90,
                })
            }

            // 6a. One-Shot Cry / Sob / Weep
            "crying" | "cry" | "cries" | "weep" | "weeping" | "sob" | "sobs" | "sobbing"
            | "sfx-cry" | "sfx-sob" | "泣き" | "泣" | "嗚咽" => {
                sliders.insert("sad".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.8),
                    arousal: Some(0.2),
                    cfg_scale: 1.2,
                    speed_factor: 0.90,
                })
            }
            // 6b. Carry Sad / Sorrow / Pleading
            "sad" | "sadly" | "sorrow" | "sorrowful" | "sorrowfully" | "mournful"
            | "mournfully" | "grief" | "grieving" | "heartbroken" | "melancholy"
            | "wistful" | "tearful" | "tearfully" | "ashamed" | "hurt" | "remorseful"
            | "plead" | "pleading" | "pleadingly" | "beg" | "begging" | "imploring"
            | "emotion-sad" | "emotion-sorrow" | "emotion-melancholy"
            | "悲しい" | "悲しみ" | "哀" | "懇願" => {
                sliders.insert("sad".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.6),
                    arousal: Some(-0.4),
                    cfg_scale: 1.2,
                    speed_factor: 0.90,
                })
            }

            // 7a. One-Shot Groan / Moan / Grunt
            "groan" | "groans" | "groaning" | "moan" | "moans" | "moaning"
            | "grunt" | "grunts" | "sfx-groan" | "sfx-moan"
            | "うめき" | "呻き" | "喘ぎ" => {
                sliders.insert("sad".to_string(), 0.5);
                sliders.insert("angry".to_string(), 0.3);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.4),
                    arousal: Some(0.2),
                    cfg_scale: 1.15,
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
                sliders.insert("angry".to_string(), 0.3);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.5),
                    arousal: Some(0.1),
                    cfg_scale: 1.15,
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
                    valence: Some(-0.4),
                    arousal: Some(-0.2),
                    cfg_scale: 1.15,
                    speed_factor: 0.92,
                })
            }

            // 9a. One-Shot Scream / Shriek
            "scream" | "screams" | "screaming" | "shriek" | "shrieking" | "sfx-scream"
            | "悲鳴" => {
                sliders.insert("angry".to_string(), 0.75);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.7),
                    arousal: Some(0.8),
                    cfg_scale: 1.2,
                    speed_factor: 1.15,
                })
            }
            // 9b. Carry Angry / Shout / Sulking / Displeased
            "angry" | "angrily" | "anger" | "mad" | "furious" | "furiously" | "enraged"
            | "irritated" | "irritatedly" | "annoyed" | "frustrated" | "frustratedly"
            | "impatient" | "impatiently" | "displeased" | "dissatisfied" | "sulking"
            | "sulk" | "sulkily" | "pouting" | "pout" | "grumpy" | "bitter" | "bitterly"
            | "shout" | "shouts" | "shouting" | "yell" | "yells" | "yelling" | "loud"
            | "loudly" | "menacing" | "menacingly" | "threatening" | "defiant" | "defiantly"
            | "stern" | "sternly" | "commanding" | "authoritative" | "gruff" | "gruffly"
            | "emotion-angry" | "emotion-furious" | "emotion-frustrated" | "style-shout"
            | "怒り" | "怒" | "叫び" | "不満" | "拗ね" => {
                sliders.insert("angry".to_string(), 0.75);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.6),
                    arousal: Some(0.6),
                    cfg_scale: 1.2,
                    speed_factor: 1.15,
                })
            }

            // 10. Carry Surprise / Shock / Questioning / Curious
            "surprise" | "surprised" | "shock" | "shocked" | "startled" | "astonished"
            | "amazed" | "awed" | "awe" | "questioning" | "questioningly" | "wondering"
            | "wonderingly" | "curious" | "curiously" | "puzzled"
            | "emotion-surprise" | "emotion-surprised" | "emotion-curious"
            | "驚き" | "驚" | "感嘆" | "疑問" => {
                sliders.insert("surprised".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.2),
                    arousal: Some(0.6),
                    cfg_scale: 1.2,
                    speed_factor: 1.0,
                })
            }

            // 11. One-Shot Gasp / Pant / Breathless
            "gasp" | "gasps" | "gasping" | "pant" | "pants" | "panting"
            | "heavy breathing" | "shortness of breath" | "breathless" | "breathlessly"
            | "out of breath" | "sfx-gasp" | "sfx-pant"
            | "ハッ" | "息をのむ" | "息切れ" => {
                sliders.insert("surprised".to_string(), 0.6);
                sliders.insert("fear".to_string(), 0.3);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.1),
                    arousal: Some(0.7),
                    cfg_scale: 1.15,
                    speed_factor: 1.0,
                })
            }

            // 12a. One-Shot Stutter / Stammer
            "stutter" | "stutters" | "stuttering" | "stammer" | "stammering"
            | "どもり" => {
                sliders.insert("fear".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::OneShot,
                    sliders,
                    valence: Some(-0.4),
                    arousal: Some(0.4),
                    cfg_scale: 1.2,
                    speed_factor: 1.0,
                })
            }
            // 12b. Carry Fear / Scared / Nervous / Shy / Worried
            "fear" | "fearful" | "scared" | "terrified" | "frightened" | "afraid"
            | "panicked" | "panic" | "agitated" | "agitatedly" | "nervous" | "nervously"
            | "anxious" | "anxiously" | "worried" | "worriedly" | "concerned" | "uneasy"
            | "trembling" | "trembling voice" | "quavering" | "shaky" | "shakily"
            | "hesitant" | "hesitantly" | "uncertain" | "uncertainly" | "shy" | "shyly"
            | "bashful" | "bashfully" | "timid" | "timidly" | "embarrassed" | "flustered"
            | "urgent" | "urgently" | "desperate" | "desperately"
            | "emotion-fear" | "emotion-anxious" | "emotion-nervous" | "emotion-panicked"
            | "emotion-worried" | "emotion-shy" | "emotion-embarrassed"
            | "恐れ" | "怖" | "恐怖" | "慌て" | "動揺" | "心配" | "照れ" => {
                sliders.insert("fear".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.5),
                    arousal: Some(0.5),
                    cfg_scale: 1.2,
                    speed_factor: 1.0,
                })
            }

            // 13. Carry Disgust
            "disgust" | "disgusted" | "emotion-disgust" | "嫌悪" => {
                sliders.insert("disgust".to_string(), 0.7);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.7),
                    arousal: Some(0.2),
                    cfg_scale: 1.2,
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
                    cfg_scale: 1.15,
                    speed_factor: 0.95,
                })
            }
            // 14b. Carry Sarcastic / Mocking / Cold / Dismissive
            "sarcastic" | "sarcastically" | "sarcasm" | "ironic" | "mocking" | "mockingly"
            | "smug" | "smugly" | "smirk" | "exasperated" | "exasperatedly"
            | "cold" | "coldly" | "grim" | "grimly" | "dismissive" | "dismissively"
            | "suspicious" | "suspiciously" | "style-sarcastic" | "style-mocking"
            | "emotion-annoyed" | "皮肉" | "呆れ" => {
                sliders.insert("angry".to_string(), 0.3);
                sliders.insert("sad".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(-0.3),
                    arousal: Some(0.0),
                    cfg_scale: 1.15,
                    speed_factor: 0.95,
                })
            }

            // 15. Carry Dramatic / Solemn / Singing
            "dramatic" | "dramatic tone" | "solemn" | "solemnly" | "echo" | "reverb" => {
                sliders.insert("surprised".to_string(), 0.3);
                sliders.insert("sad".to_string(), 0.2);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.0),
                    arousal: Some(0.2),
                    cfg_scale: 1.2,
                    speed_factor: 0.90,
                })
            }
            "advertisement" | "commercial" | "sings" | "singing" | "hum" | "humming"
            | "style-singing" | "style-hum" | "鼻歌" => {
                sliders.insert("happy".to_string(), 0.4);
                Some(EmotionProfile {
                    layer: TagLayer::Carry,
                    sliders,
                    valence: Some(0.5),
                    arousal: Some(0.3),
                    cfg_scale: 1.15,
                    speed_factor: 1.05,
                })
            }

            // 16. Reset / Calm / Neutral
            "narration" | "narrator" | "calm" | "calmly" | "neutral" | "normal"
            | "serious" | "seriously" | "thoughtful" | "thoughtfully" | "determined"
            | "relieved" | "satisfied" | "contentedly" | "deadpan" | "monotone"
            | "confused" | "emotion-calm" | "emotion-relieved" | "style-narration"
            | "安堵" | "満足" | "落ち着き" => Some(EmotionProfile {
                layer: TagLayer::Reset,
                sliders,
                valence: None,
                arousal: None,
                cfg_scale: 1.0,
                speed_factor: 1.0,
            }),

            // 17. Carry Pacing Modifiers
            "slowly" | "slow" | "deliberate" | "deliberately" | "prosody-slow" | "ゆっくり" => {
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
            | "hurriedly" | "prosody-fast" | "早口" => Some(EmotionProfile {
                layer: TagLayer::Carry,
                sliders,
                valence: None,
                arousal: None,
                cfg_scale: 1.0,
                speed_factor: 1.20,
            }),

            // 18. Pure sound/action tags (stripped from text without altering emotion)
            "clear throat" | "clears throat" | "clearing throat" | "cough" | "coughs"
            | "coughing" | "pause" | "silence" | "sneeze" | "sneezes" | "sneezing"
            | "yawn" | "yawns" | "yawning" | "gulp" | "gulps" | "swallow" | "swallows"
            | "tsk" | "tut" | "click tongue" | "tongue click" | "lip smack" | "lip noise"
            | "kiss" | "smooch" | "lick" | "licking" | "chew" | "chewing" | "wet sound"
            | "nod" | "agree" | "phone" | "telephone" => Some(EmotionProfile {
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

    pub fn parse(input: &str) -> ParsedPrompt {
        let mut detected_tags = Vec::new();

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

        // 5. Classify tags into Carry and One-Shot layers and blend
        let mut active_carries: Vec<EmotionProfile> = Vec::new();
        let mut active_oneshots: Vec<EmotionProfile> = Vec::new();

        for tag in &detected_tags {
            let base_tag = Self::strip_modifier_prefix(tag.as_str());
            if let Some(profile) = Self::classify_tag(base_tag) {
                match profile.layer {
                    TagLayer::Carry => active_carries.push(profile),
                    TagLayer::OneShot => active_oneshots.push(profile),
                    TagLayer::Reset => {
                        active_carries.clear();
                        active_oneshots.clear();
                    }
                    TagLayer::Neutral => {}
                }
            }
        }

        let carry_merged = match active_carries.as_slice() {
            [] => None,
            [single] => Some(single.clone()),
            [first, .., last] => Some(Self::blend_profiles(
                first,
                Self::BASE_CARRY_WEIGHT,
                last,
                Self::OVERRIDE_CARRY_WEIGHT,
            )),
        };

        let oneshot_merged = active_oneshots.last().cloned();

        let final_profile = match (carry_merged, oneshot_merged) {
            (Some(carry), Some(oneshot)) => Some(Self::blend_profiles(
                &carry,
                Self::CARRY_BLEND_WEIGHT,
                &oneshot,
                Self::ONESHOT_BLEND_WEIGHT,
            )),
            (Some(carry), None) => Some(carry),
            (None, Some(oneshot)) => Some(oneshot),
            (None, None) => None,
        };

        if let Some(profile) = final_profile {
            let emotion_enabled = !profile.sliders.is_empty();
            ParsedPrompt {
                cleaned_text: cleaned_trimmed,
                emotion_enabled,
                emotion_sliders: profile.sliders,
                emotion_valence: if emotion_enabled { profile.valence } else { None },
                emotion_arousal: if emotion_enabled { profile.arousal } else { None },
                emotion_cfg_scale: profile.cfg_scale,
                speed_factor: profile.speed_factor,
                accurate_mode: !emotion_enabled,
                detected_tags,
            }
        } else {
            ParsedPrompt {
                cleaned_text: cleaned_trimmed,
                emotion_enabled: false,
                emotion_sliders: HashMap::new(),
                emotion_valence: None,
                emotion_arousal: None,
                emotion_cfg_scale: 1.0,
                speed_factor: 1.0,
                accurate_mode: true,
                detected_tags,
            }
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
        assert!(parsed.emotion_enabled);
        assert!(!parsed.accurate_mode);
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
        assert!(parsed.emotion_enabled);
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
        assert!(!parsed.emotion_enabled);
        assert!(parsed.accurate_mode);
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
        // cfg_scale: 1.1 * 0.35 + 1.2 * 0.65 = 1.165
        assert_eq!(parsed.emotion_cfg_scale, 1.165);
    }

    #[test]
    fn test_reset_tag_clears_previous_carry() {
        let input = "[whisper] 秘密だよ… [calm] さて、普通の話に戻ろうか。";
        let parsed = EmotionParser::parse(input);
        assert_eq!(parsed.cleaned_text, "秘密だよ… さて、普通の話に戻ろうか。");
        assert!(!parsed.emotion_enabled);
        assert!(parsed.emotion_sliders.is_empty());
        assert_eq!(parsed.speed_factor, 1.0);
        assert!(parsed.accurate_mode);
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
