use regex::Regex;
use std::sync::LazyLock;

/// Regex matching `<SE>` and `</SE>` wrapper tags used by Higgs/OpenAI audio prompts
static SE_WRAPPER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)</?se>")
        .expect("Failed to compile SE_WRAPPER_REGEX in irodori mapper")
});

/// Regex matching full bracketed or starred tags:
/// [tag], 【tag】, (tag), （tag）, *tag*
static FULL_TAG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(?P<tag>[^\]]+)\]|【(?P<ztag>[^】]+)】|\((?P<ptag>[^)]+)\)|（(?P<fptag>[^）]+)）|\*(?P<atag>[^*]+)\*")
        .expect("Failed to compile FULL_TAG_REGEX in irodori mapper")
});

/// Missing opening bracket: e.g. "chuckle] Hello", " clear throat] text"
static MISSING_OPEN_BRACKET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?P<prefix>^|[\s.,!?。、！？])(?P<otag>[a-z][a-z0-9_\- ]{1,25})\]")
        .expect("Failed to compile MISSING_OPEN_BRACKET_REGEX in irodori mapper")
});

/// Missing closing bracket: e.g. "[whisper Hello", " [sigh text"
static MISSING_CLOSE_BRACKET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\[(?P<itag>[a-z][a-z0-9_\- ]{1,25})(?P<suffix>[\s.,!?。、！？]|$)")
        .expect("Failed to compile MISSING_CLOSE_BRACKET_REGEX in irodori mapper")
});

/// Cleans up orphan bracket or asterisk symbols from text boundaries
static ORPHAN_BRACKETS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[\[\]【】()（）*]+|[\[\]【】()（）*]+$")
        .expect("Failed to compile ORPHAN_BRACKETS_REGEX in irodori mapper")
});

static MULTI_SPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[ \t]{2,}")
        .expect("Failed to compile MULTI_SPACE_REGEX in irodori mapper")
});

#[derive(Debug, Clone, PartialEq)]
pub struct IrodoriMappedText {
    /// Text with tags replaced by Irodori-compatible emojis
    pub prompt_text: String,
    /// Detected tags in original order
    pub detected_tags: Vec<String>,
}

pub struct IrodoriMapper;

impl IrodoriMapper {
    /// Maps a raw tag string to an official Irodori-TTS emoji (all 36 official styles),
    /// covering all 138 SkyrimNet carry_tags_*.yaml tags, 0650_audio_tags.prompt tags,
    /// and Higgs-style tags, or None if the tag should be safely removed without vocalization.
    pub fn tag_to_emoji(tag: &str) -> Option<&'static str> {
        let normalized = tag.trim().to_lowercase();
        match normalized.as_str() {
            // === 1. Whispering & Soft Ear-level Voice (👂) ===
            "whisper" | "whispering" | "whispers" | "whispered" | "hushed" | "quiet"
            | "quietly" | "shush" | "shh" | "murmurs" | "murmuring" | "mutters"
            | "muttering" | "mumbles" | "mumbling" | "conspiratorial" | "style-whisper"
            | "ささやき" | "囁き" | "耳元" | "小声" | "静かに" => Some("👂"),

            // === 2. Gentle & Tender & Warm & Seductive (🫶) ===
            "gentle" | "gently" | "sweet" | "soft" | "softly" | "tender" | "tenderly"
            | "warm" | "warmly" | "seductive" | "seductively" | "sultry"
            | "flirtatious" | "flirtatiously" | "reverent"
            | "優しく" | "慈しむ" => Some("🫶"),

            // === 3. Sigh & Breathing & Tired & Resigned (😮‍💨) ===
            "sigh" | "sighs" | "sighing" | "exhale" | "inhale"
            | "tired" | "exhausted" | "weary" | "wearily" | "resigned" | "bored"
            | "sfx-sigh" | "sfx-breath"
            | "ため息" | "息" | "寝息" => Some("😮‍💨"),

            // === 4. Shortness of breath & Panting (🌬️) ===
            "pant" | "pants" | "panting" | "breathless" | "breathlessly" | "out of breath"
            | "sfx-pant"
            | "息切れ" | "激しい呼吸" => Some("🌬️"),

            // === 5. Chuckling & Suppressed Giggling (🤭) ===
            "chuckle" | "chuckles" | "chuckling" | "giggle" | "giggles" | "giggling"
            | "snicker" | "snickers" | "sfx-chuckle" | "sfx-giggle" | "style-chuckle"
            | "style-giggle"
            | "くすくす" | "忍び笑い" | "吹き出し" => Some("🤭"),

            // === 6. Happy & Joy & Bright Smile & Proud (😊) ===
            "happy" | "happily" | "joy" | "joyful" | "joyfully" | "cheerful" | "cheerfully"
            | "smile" | "hopeful" | "proud" | "proudly" | "confident" | "confidently"
            | "emotion-happy" | "emotion-joy" | "emotion-cheerful" | "emotion-proud"
            | "喜" | "歓喜" | "明るく" | "嬉しそう" => Some("😊"),

            // === 7. Excited & Exuberant Laughter (😆) ===
            "laugh" | "laughs" | "laughter" | "laughing" | "excited" | "excitedly"
            | "sfx-laughter" | "sfx-laugh" | "emotion-excited"
            | "笑い" | "大笑い" | "笑" | "わくわく" | "興奮" | "楽しそう" => Some("😆"),

            // === 8. Angry & Screaming & Yelling & Stern (😡) ===
            "angry" | "angrily" | "anger" | "mad" | "furious" | "furiously"
            | "irritated" | "frustrated" | "impatient" | "impatiently"
            | "bitter" | "bitterly" | "disgusted" | "disgust"
            | "shout" | "shouts" | "shouting" | "yell" | "yells" | "yelling"
            | "loudly" | "scream" | "screams" | "screaming"
            | "menacing" | "menacingly" | "threatening" | "defiant" | "defiantly"
            | "stern" | "sternly" | "commanding" | "authoritative" | "gruff" | "gruffly"
            | "emotion-angry" | "emotion-furious" | "emotion-frustrated" | "emotion-disgust"
            | "style-shout"
            | "怒り" | "怒" | "叫び" | "不満" | "拗ね" => Some("😡"),

            // === 9. Sad & Crying & Weeping & Sorrow (😭) ===
            "sad" | "sadly" | "sorrow" | "sorrowful" | "mournful" | "grief" | "grieving"
            | "melancholy" | "wistful" | "tearful" | "tearfully" | "ashamed"
            | "crying" | "cry" | "cries" | "weep" | "weeping" | "sob" | "sobs" | "sobbing"
            | "emotion-sad" | "emotion-sorrow" | "emotion-melancholy" | "sfx-cry" | "sfx-sob"
            | "悲しい" | "哀" | "泣き" | "泣" | "すすり泣き" => Some("😭"),

            // === 10. Trembling Voice & Timid & Hesitant (🥺) ===
            "trembling" | "timid" | "timidly" | "shaky" | "shakily"
            | "weak" | "weakly" | "hesitant" | "hesitantly"
            | "おどおど" | "震え声" | "頼りない" => Some("🥺"),

            // === 11. Fear & Panicked & Urgent & Desperate (😰) ===
            "fear" | "fearful" | "scared" | "terrified" | "frightened" | "afraid" | "panicked"
            | "nervous" | "nervously" | "anxious" | "anxiously"
            | "urgent" | "urgently" | "desperate" | "desperately"
            | "emotion-fear" | "emotion-anxious" | "emotion-nervous" | "emotion-panicked"
            | "恐れ" | "怖" | "恐怖" | "慌て" | "焦り" | "取り乱す" => Some("😰"),

            // === 12. Surprise & Shock & Gasp (😲) ===
            "surprise" | "surprised" | "shock" | "shocked" | "startled" | "astonished"
            | "gasp" | "gasps" | "gasping" | "emotion-surprise" | "emotion-surprised" | "sfx-gasp"
            | "驚き" | "驚" | "ハッ" | "息をのむ" | "感嘆" => Some("😲"),

            // === 13. Distressed & In Pain & Strained (😖) ===
            "distress" | "pain" | "pained" | "strained" | "hoarse" | "hoarsely"
            | "wince" | "winces" | "emotion-pain"
            | "苦しい" | "苦しそう" => Some("😖"),

            // === 14. Worried & Uneasy (😟) ===
            "worried" | "uneasy" | "concerned" | "emotion-worried"
            | "心配" | "不安" => Some("😟"),

            // === 15. Annoyed & Exasperated & Cold & Dismissive (🙄) ===
            "annoyed" | "exasperated" | "dismissive" | "dismissively"
            | "cold" | "coldly" | "grim" | "grimly" | "deadpan" | "monotone"
            | "emotion-annoyed"
            | "呆れ" | "やれやれ" => Some("🙄"),

            // === 16. Tongue Clicking (😒) ===
            "click tongue" | "tongue click" | "tsk" | "tut" | "sfx-tsk"
            | "舌打ち" => Some("😒"),

            // === 17. Sarcastic & Teasing & Smirk & Playful (😏) ===
            "sarcastic" | "sarcastically" | "sarcasm" | "ironic"
            | "mocking" | "mockingly" | "smug" | "smugly" | "smirk" | "scoff" | "scoffs"
            | "playful" | "playfully" | "teasing" | "teasingly"
            | "style-sarcastic" | "style-teasing" | "style-playful" | "style-mocking"
            | "皮肉" | "からかい" | "甘え" | "宥め" | "ニヤリ" => Some("😏"),

            // === 18. Shy & Bashful & Embarrassed (🫣) ===
            "shy" | "shyly" | "bashful" | "embarrassed" | "flustered"
            | "emotion-shy" | "emotion-embarrassed"
            | "恥ずかしい" | "照れ" => Some("🫣"),

            // === 19. Relieved & Satisfied & Calm & Solemn (😌) ===
            "relieved" | "satisfied" | "calm" | "calmly" | "serious" | "seriously"
            | "solemn" | "solemnly" | "determined" | "narration" | "narrator"
            | "emotion-calm" | "emotion-relieved" | "emotion-serious" | "style-narration"
            | "安堵" | "ほっとした" | "満足" => Some("😌"),

            // === 20. Sleepy & Sluggish (😴) ===
            "sleepy" | "sleepily" | "drowsy"
            | "眠い" | "気だるげ" => Some("😴"),

            // === 21. Yawn (🥱) ===
            "yawn" | "yawns" | "yawning" | "sfx-yawn"
            | "あくび" => Some("🥱"),

            // === 22. Plead & Beg (🙏) ===
            "plead" | "pleading" | "beg" | "begging" | "imploring"
            | "お願い" | "頼み" => Some("🙏"),

            // === 23. Drunk (🥴) ===
            "drunk" | "drunkenly" | "tipsy" | "slurred"
            | "酔っ払い" | "泥酔" => Some("🥴"),

            // === 24. Wondering & Questioning & Thoughtful & Curious (🤔) ===
            "wondering" | "questioning" | "thoughtful" | "thoughtfully"
            | "curious" | "curiously" | "confused" | "suspicious" | "suspiciously"
            | "pondering" | "emotion-curious" | "emotion-confused"
            | "疑問" | "思案" => Some("🤔"),

            // === 25. Agreeing / Backchanneling (👌) ===
            "agree" | "nod" | "相槌" | "肯定" => Some("👌"),

            // === 26. Groaning & Moaning & Heavy Breathing (🥵) ===
            "groan" | "groans" | "groaning" | "moan" | "moans" | "moaning"
            | "grunt" | "grunts" | "sfx-groan" | "sfx-moan" | "sfx-grunt"
            | "うめき" | "呻き" | "喘ぎ" | "息遣い" => Some("🥵"),

            // === 27. Coughing & Throat-clearing & Sneezing & Sniffling (🤧) ===
            "clear throat" | "clears throat" | "clearing throat" | "throat-clearing"
            | "cough" | "coughs" | "coughing"
            | "sniff" | "sniffs" | "sniffling" | "sniffle" | "sniffles"
            | "sneeze" | "sneezes" | "sneezing"
            | "sfx-cough" | "sfx-clear-throat" | "sfx-sneeze" | "sfx-sniff"
            | "咳" | "咳払い" | "くしゃみ" | "鼻をすする" => Some("🤧"),

            // === 28. Wet sounds & Licking & Chewing (👅) ===
            "lick" | "licking" | "chew" | "wet sound" | "舌舐めずり" | "咀嚼音" | "水音" => Some("👅"),

            // === 29. Lip smack & Lip noise & Kiss (💋) ===
            "lip smack" | "lip noise" | "kiss" | "smooch" | "sfx-kiss"
            | "リップノイズ" | "チュッ" => Some("💋"),

            // === 30. Gulping & Swallowing (🥤) ===
            "gulp" | "gulps" | "swallow" | "swallows" | "sfx-gulp" | "sfx-swallow"
            | "飲み込む" | "ゴクリ" => Some("🥤"),

            // === 31. Loudspeaker & Reverb / Echo & Advertisement (📢) ===
            "dramatic" | "dramatic tone" | "advertisement" | "commercial" | "announcer"
            | "loudspeaker" | "megaphone" | "echo" | "reverb"
            | "拡声器" | "エコー" | "リバーブ" => Some("📢"),

            // === 32. Over Phone / Speaker (📞) ===
            "phone" | "telephone" | "電話" | "スピーカー越し" => Some("📞"),

            // === 33. Muffled Voice (🤐) ===
            "muffled" | "こもった声" => Some("🤐"),

            // === 34. Humming & Singing (🎵) ===
            "humming" | "hum" | "sings" | "singing" | "sing" | "song"
            | "style-singing" | "style-hum"
            | "鼻歌" => Some("🎵"),

            // === 35. Pause & Silence (⏸️) ===
            "pause" | "silence" | "beat" | "hesitation"
            | "間" | "沈黙" => Some("⏸️"),

            // === 36. Speed: Slowly (🐢) & Fast (⏩) ===
            "slowly" | "slow" | "deliberate" | "deliberately" | "prosody-slow"
            | "ゆっくり" => Some("🐢"),
            "fast" | "quickly" | "quick" | "rushed" | "hurried" | "hurriedly"
            | "rapidly" | "prosody-fast"
            | "早口" | "急ぎ" => Some("⏩"),

            // Unknown tags: safely remove so they are never spoken aloud
            _ => None,
        }
    }

    /// Converts raw text containing audio/emotion tags into Irodori emoji-embedded text.
    pub fn convert(input: &str) -> IrodoriMappedText {
        let mut detected_tags = Vec::new();

        // 0. Strip <SE> / </SE> wrapper tags if present
        let input_no_se = SE_WRAPPER_REGEX.replace_all(input, " ");

        // 1. Full enclosed tags: [tag], (tag), *tag*, etc.
        let text_after_full = FULL_TAG_REGEX.replace_all(&input_no_se, |caps: &regex::Captures| {
            let raw_tag = caps
                .name("tag")
                .or_else(|| caps.name("ztag"))
                .or_else(|| caps.name("ptag"))
                .or_else(|| caps.name("fptag"))
                .or_else(|| caps.name("atag"))
                .map(|m| m.as_str().trim().to_lowercase())
                .unwrap_or_default();

            if !raw_tag.is_empty() {
                detected_tags.push(raw_tag.clone());
            }

            match Self::tag_to_emoji(&raw_tag) {
                Some(emoji) => format!(" {emoji} "),
                None => " ".to_string(),
            }
        });

        // 2. Rescue tags missing opening bracket (e.g. "chuckle] Hello")
        let text_after_missing_open = MISSING_OPEN_BRACKET_REGEX.replace_all(&text_after_full, |caps: &regex::Captures| {
            let prefix = caps.name("prefix").map(|m| m.as_str()).unwrap_or("");
            let raw_tag = caps.name("otag").map(|m| m.as_str().trim().to_lowercase()).unwrap_or_default();

            if !raw_tag.is_empty() {
                detected_tags.push(raw_tag.clone());
            }

            match Self::tag_to_emoji(&raw_tag) {
                Some(emoji) => format!("{prefix} {emoji} "),
                None => format!("{prefix} "),
            }
        });

        // 3. Rescue tags missing closing bracket (e.g. "[whisper Hello")
        let text_after_missing_close = MISSING_CLOSE_BRACKET_REGEX.replace_all(&text_after_missing_open, |caps: &regex::Captures| {
            let suffix = caps.name("suffix").map(|m| m.as_str()).unwrap_or("");
            let raw_tag = caps.name("itag").map(|m| m.as_str().trim().to_lowercase()).unwrap_or_default();

            if !raw_tag.is_empty() {
                detected_tags.push(raw_tag.clone());
            }

            match Self::tag_to_emoji(&raw_tag) {
                Some(emoji) => format!(" {emoji} {suffix}"),
                None => format!(" {suffix}"),
            }
        });

        // 4. Sanitize whitespace and boundary orphan brackets
        let collapsed = MULTI_SPACE_REGEX.replace_all(&text_after_missing_close, " ").to_string();
        let trimmed = collapsed.trim();
        let sanitized = ORPHAN_BRACKETS_REGEX.replace_all(trimmed, "").to_string();
        let prompt_text = sanitized.trim().to_string();

        IrodoriMappedText {
            prompt_text,
            detected_tags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_irodori_mapper_tags() {
        let input = "[whisper] パパ、寒くない？ [sigh]";
        let mapped = IrodoriMapper::convert(input);
        assert_eq!(mapped.detected_tags, vec!["whisper", "sigh"]);
        assert!(mapped.prompt_text.contains("👂"));
        assert!(mapped.prompt_text.contains("😮‍💨"));
        assert!(mapped.prompt_text.contains("パパ、寒くない？"));
    }

    #[test]
    fn test_irodori_mapper_missing_brackets() {
        let input = "chuckle] さあケンジ君、やっとウィンターホールドに着いたわ。";
        let mapped = IrodoriMapper::convert(input);
        assert_eq!(mapped.detected_tags, vec!["chuckle"]);
        assert!(mapped.prompt_text.contains("🤭"));
        assert!(mapped.prompt_text.contains("さあケンジ君、やっとウィンターホールドに着いたわ。"));
        assert!(!mapped.prompt_text.contains("chuckle]"));
    }

    #[test]
    fn test_irodori_mapper_throat_clearing_to_sneeze_emoji() {
        let input = "[clear throat] でも、馬車の中なら変な輩に邪魔される心配もないわね。";
        let mapped = IrodoriMapper::convert(input);
        assert_eq!(mapped.detected_tags, vec!["clear throat"]);
        assert!(mapped.prompt_text.contains("🤧"));
        assert!(!mapped.prompt_text.contains("clear throat"));
        assert!(mapped.prompt_text.contains("でも、馬車の中なら変な輩に邪魔される心配もないわね。"));
    }

    #[test]
    fn test_irodori_mapper_unknown_tag_stripping() {
        let input = "[unknown_tag_123] こんにちは、ケンジ君。";
        let mapped = IrodoriMapper::convert(input);
        assert_eq!(mapped.detected_tags, vec!["unknown_tag_123"]);
        assert!(!mapped.prompt_text.contains("unknown_tag_123"));
        assert_eq!(mapped.prompt_text, "こんにちは、ケンジ君。");
    }

    #[test]
    fn test_irodori_mapper_parentheses_and_asterisks() {
        let input = "(sigh) イリアさん、どうかその魔法を収めて！ *giggle*";
        let mapped = IrodoriMapper::convert(input);
        assert_eq!(mapped.detected_tags, vec!["sigh", "giggle"]);
        assert!(mapped.prompt_text.contains("😮‍💨"));
        assert!(mapped.prompt_text.contains("🤭"));
    }

    #[test]
    fn test_irodori_official_styles_coverage() {
        // Test key official styles: breathing, panting, gentle, excited, pain, throat clearing, etc.
        assert_eq!(IrodoriMapper::tag_to_emoji("pant"), Some("🌬️"));
        assert_eq!(IrodoriMapper::tag_to_emoji("gentle"), Some("🫶"));
        assert_eq!(IrodoriMapper::tag_to_emoji("excited"), Some("😆"));
        assert_eq!(IrodoriMapper::tag_to_emoji("groan"), Some("🥵"));
        assert_eq!(IrodoriMapper::tag_to_emoji("fear"), Some("😰"));
        assert_eq!(IrodoriMapper::tag_to_emoji("clear throat"), Some("🤧"));
        assert_eq!(IrodoriMapper::tag_to_emoji("dramatic"), Some("📢"));
    }

    #[test]
    fn test_irodori_skyrimnet_vbeta27_carry_tags_and_se_wrappers() {
        // Verify carry_tags_*.yaml tags and <SE>[tag]</SE> wrapper stripping
        let input = "[conspiratorial] <SE>[Laughter]</SE> [hurriedly] こっちよ、急いで！ [resigned]";
        let mapped = IrodoriMapper::convert(input);
        assert_eq!(
            mapped.detected_tags,
            vec!["conspiratorial", "laughter", "hurriedly", "resigned"]
        );
        assert!(mapped.prompt_text.contains("👂"));
        assert!(mapped.prompt_text.contains("😆"));
        assert!(mapped.prompt_text.contains("⏩"));
        assert!(mapped.prompt_text.contains("😮‍💨"));
        assert!(!mapped.prompt_text.contains("<SE>"));
        assert!(!mapped.prompt_text.contains("</SE>"));
    }
}
