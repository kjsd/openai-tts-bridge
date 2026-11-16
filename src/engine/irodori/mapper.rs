use crate::parser::emotion::TagLayer;
use crate::parser::EmotionParser;
use regex::Regex;
use std::sync::LazyLock;

/// Regex matching `<SE>` and `</SE>` wrapper tags used by Higgs/OpenAI audio prompts
static SE_WRAPPER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)</?se>").expect("Failed to compile SE_WRAPPER_REGEX in irodori mapper")
});

/// Regex matching full bracketed or starred tags:
/// [tag], 【tag】, (tag), （tag）, *tag*
static FULL_TAG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(?P<tag>[^\]]+)\]|【(?P<ztag>[^】]+)】|\((?P<ptag>[^)]+)\)|（(?P<fptag>[^）]+)）|\*(?P<atag>[^*]+)\*")
        .expect("Failed to compile FULL_TAG_REGEX in irodori mapper")
});

/// Missing opening bracket: e.g. "chuckle] Hello", " clear throat] text"
static MISSING_OPEN_BRACKET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?P<prefix>^|[\s.,!?。、！？])(?P<otag>[a-z][a-z0-9_\-: ]{1,30})\]")
        .expect("Failed to compile MISSING_OPEN_BRACKET_REGEX in irodori mapper")
});

/// Missing closing bracket: e.g. "[whisper Hello", " [sigh text"
static MISSING_CLOSE_BRACKET_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\[(?P<itag>[a-z][a-z0-9_\-: ]{1,30})(?P<suffix>[\s.,!?。、！？]|$)")
        .expect("Failed to compile MISSING_CLOSE_BRACKET_REGEX in irodori mapper")
});

/// Cleans up orphan bracket or asterisk symbols from text boundaries
static ORPHAN_BRACKETS_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[\[\]【】()（）*]+|[\[\]【】()（）*]+$")
        .expect("Failed to compile ORPHAN_BRACKETS_REGEX in irodori mapper")
});

static MULTI_SPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[ \t]{2,}").expect("Failed to compile MULTI_SPACE_REGEX in irodori mapper")
});

static SPEAKABLE_CHAR_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{Hiragana}\p{Katakana}\p{Han}a-zA-Z0-9]")
        .expect("Failed to compile SPEAKABLE_CHAR_REGEX in irodori mapper")
});

/// Intensifier prefixes that double the mapped emoji for stronger effect
/// per Irodori-TTS EMOJI_ANNOTATIONS.md ("emphasize the effect by using the same emoji multiple times").
const INTENSIFIER_PREFIXES: &[&str] = &[
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
];

/// Softener prefixes stripped when resolving a base tag
const SOFTENER_PREFIXES: &[&str] = &[
    "slightly ",
    "mildly ",
    "a bit ",
    "a little ",
    "somewhat ",
    "soft ",
    "softly ",
    "quiet ",
    "quietly ",
];

#[derive(Debug, Clone, PartialEq)]
pub struct IrodoriMappedText {
    /// Text with tags replaced by Irodori-compatible emojis
    pub prompt_text: String,
    /// Detected tags in original order
    pub detected_tags: Vec<String>,
}

pub struct IrodoriMapper;

impl IrodoriMapper {
    /// Maps a raw tag string to one of the 39 official Aratako/Irodori-TTS-500M emojis
    /// (`EMOJI_ANNOTATIONS.md`), covering all SkyrimNet `carry_tags_*.yaml` tags,
    /// `0650_audio_tags.prompt` tags, `nina_agent` tags, and Higgs-style tags,
    /// or `None` if the tag should be safely removed without vocalization (such as `[calm]` / `[reset]`).
    pub fn tag_to_emoji(tag: &str) -> Option<&'static str> {
        let normalized = tag.trim().to_lowercase();
        match normalized.as_str() {
            // === Reset / Neutral tags: strip cleanly without emitting an emoji ===
            "calm" | "calmly" | "neutral" | "normal" | "default" | "reset" | "emotion-calm" => None,

            // === 1. 👂 : 囁き、耳元の音 (Whisper, sounds close to the ear) ===
            "👂" | "whisper" | "whispering" | "whispers" | "whispered" | "hushed" | "quiet"
            | "quietly" | "shush" | "shh" | "murmur" | "murmurs" | "murmuring" | "mutter"
            | "mutters" | "muttering" | "mumble" | "mumbles" | "mumbling" | "conspiratorial"
            | "conspiratorially" | "under breath" | "faint" | "faintly" | "soft voice"
            | "close to ear" | "style-whisper" | "style-whispering"
            | "ささやき" | "囁き" | "耳元" | "耳元の音" | "小声" | "静かに" => Some("👂"),

            // === 2. 😮‍💨 : 吐息、溜息、寝息 (Breath, sigh, sleeping breath) ===
            "😮‍💨" | "sigh" | "sighs" | "sighing" | "breath" | "breathe" | "breathing"
            | "exhale" | "exhales" | "exhaling" | "inhale" | "inhales" | "inhaling"
            | "sleeping breath" | "tired" | "tiredly" | "exhausted" | "weary" | "wearily"
            | "world-weary" | "resigned" | "resignedly" | "defeated" | "defeatedly" | "bored"
            | "boredly" | "sfx-sigh" | "sfx-breath"
            | "ため息" | "溜息" | "吐息" | "息" | "寝息" => Some("😮‍💨"),

            // === 3. ⏸️ : 間、沈黙 (Pause, silence) ===
            "⏸️" | "⏸" | "pause" | "pauses" | "paused" | "silence" | "silent" | "beat"
            | "hesitation" | "short pause" | "long pause"
            | "間" | "沈黙" => Some("⏸️"),

            // === 4. 🤭 : 笑い（くすくす、含み笑いなど） (Chuckle, giggle, suppressed laugh) ===
            "🤭" | "chuckle" | "chuckles" | "chuckling" | "giggle" | "giggles" | "giggling"
            | "snicker" | "snickers" | "snickering" | "titter" | "suppressed laugh"
            | "stifled laugh" | "amused" | "amusedly" | "sfx-chuckle" | "sfx-giggle"
            | "style-chuckle" | "style-giggle"
            | "くすくす" | "含み笑い" | "忍び笑い" | "吹き出し" => Some("🤭"),

            // === 5. 🥵 : 喘ぎ、うめき声、唸り声 (Panting, moan, groan) ===
            "🥵" | "moan" | "moans" | "moaning" | "groan" | "groans" | "groaning"
            | "grunt" | "grunts" | "grunting" | "growl" | "growls" | "growling"
            | "snarl" | "snarls" | "snarling" | "whimper" | "whimpers" | "whimpering"
            | "sensual moan" | "pleasure" | "sfx-groan" | "sfx-moan" | "sfx-grunt"
            | "喘ぎ" | "喘ぎ声" | "うめき" | "うめき声" | "呻き" | "唸り" | "唸り声" | "息遣い" => Some("🥵"),

            // === 6. 📢 : エコー、リバーブ (Echo, reverb) ===
            "📢" | "echo" | "echoing" | "reverb" | "reverberating" | "loudspeaker"
            | "megaphone" | "booming" | "bellow" | "bellows" | "bellowing" | "dramatic"
            | "dramatic tone" | "advertisement" | "commercial" | "announcer"
            | "拡声器" | "エコー" | "リバーブ" => Some("📢"),

            // === 7. 😏 : からかうように、甘えるように (Teasing, playfully sweet / coaxing) ===
            "😏" | "teasing" | "teasingly" | "tease" | "coaxing" | "coaxingly" | "coax"
            | "playful" | "playfully" | "sassy" | "sweetly" | "spoiled" | "coy" | "coyly"
            | "flirtatious" | "flirtatiously" | "flirty" | "seductive" | "seductively"
            | "sultry" | "sarcastic" | "sarcastically" | "sarcasm" | "sardonic" | "ironic"
            | "ironically" | "wry" | "wryly" | "sly" | "slyly" | "smug" | "smugly" | "smirk"
            | "smirking" | "mocking" | "mockingly" | "scoff" | "scoffs" | "scoffing" | "sneer"
            | "sneering" | "patronizing" | "patronizingly"
            | "style-sarcastic" | "style-teasing" | "style-playful" | "style-mocking"
            | "からかい" | "からかう" | "からかうように" | "甘え" | "甘える" | "甘えるように"
            | "宥め" | "皮肉" | "ニヤリ" => Some("😏"),

            // === 8. 🥺 : 声を震わせながら、自信のなさげに (Trembling voice, timidly / uncertainly) ===
            "🥺" | "trembling" | "trembling voice" | "quavering" | "quivering" | "timid"
            | "timidly" | "uncertain" | "uncertainly" | "unsure" | "insecure" | "insecurely"
            | "shaky" | "shakily" | "weak" | "weakly" | "hesitant" | "hesitantly"
            | "vulnerable" | "vulnerably" | "meek" | "meekly"
            | "おどおど" | "震え声" | "声を震わせながら" | "自信のなさげに" | "自信なさげ" | "頼りない" => Some("🥺"),

            // === 9. 🌬️ : 息切れ、荒い息遣い、呼吸音 (Shortness of breath, heavy breathing) ===
            "🌬️" | "🌬" | "pant" | "pants" | "panting" | "heavy breathing"
            | "shortness of breath" | "breathless" | "breathlessly" | "out of breath"
            | "winded" | "wheeze" | "wheezing" | "sfx-pant"
            | "息切れ" | "荒い息" | "荒い息遣い" | "激しい呼吸" | "呼吸音" => Some("🌬️"),

            // === 10. 😮 : 息をのむ (Gasp) ===
            "😮" | "gasp" | "gasps" | "gasping" | "sharp intake of breath" | "catch breath"
            | "sfx-gasp"
            | "息をのむ" | "息を呑む" | "ハッ" => Some("😮"),

            // === 11. 👅 : 舐める音、咀嚼音、水音 (Licking sound, chewing sound, wet sound) ===
            "👅" | "lick" | "licks" | "licking" | "licking sound" | "chew" | "chews"
            | "chewing" | "chewing sound" | "munch" | "munching" | "slurp" | "slurping"
            | "wet" | "wet sound" | "sfx-lick" | "sfx-chew"
            | "舐める" | "舐める音" | "舌舐めずり" | "咀嚼" | "咀嚼音" | "水音" => Some("👅"),

            // === 12. 💋 : リップノイズ (Lip smack / lip noise) ===
            "💋" | "lip smack" | "lip smacks" | "lip noise" | "smack lips" | "kiss" | "kisses"
            | "kissing" | "smooch" | "mwah" | "chu" | "sfx-kiss"
            | "リップノイズ" | "キス" | "チュッ" => Some("💋"),

            // === 13. 🫶 : 優しく (Gently, tenderly) ===
            "🫶" | "gentle" | "gently" | "tender" | "tenderly" | "soft" | "softly" | "warm"
            | "warmly" | "kind" | "kindly" | "affectionate" | "affectionately" | "loving"
            | "lovingly" | "lovely" | "caring" | "caringly" | "soothing" | "soothingly"
            | "comforting" | "comfortingly" | "reverent" | "reverently" | "sweet"
            | "emotion-affection" | "優しく" | "慈しむ" | "穏やか" => Some("🫶"),

            // === 14. 😭 : 嗚咽、泣き声、悲しみ (Sobbing, crying, sorrowfully / sadly) ===
            "😭" | "sad" | "sadly" | "sorrow" | "sorrowful" | "sorrowfully" | "mournful"
            | "mournfully" | "grief" | "grieving" | "heartbroken" | "devastated"
            | "melancholy" | "melancholic" | "wistful" | "wistfully" | "tearful" | "tearfully"
            | "hurt" | "remorseful" | "remorsefully" | "regretful" | "regretfully" | "ashamed"
            | "guilty" | "cry" | "cries" | "crying" | "weep" | "weeps" | "weeping"
            | "sob" | "sobs" | "sobbing" | "wail" | "wailing" | "bawl" | "bawling"
            | "emotion-sad" | "emotion-sadness" | "emotion-sorrow" | "emotion-melancholy"
            | "sfx-cry" | "sfx-crying" | "sfx-sob"
            | "悲しい" | "悲しみ" | "哀" | "泣き" | "泣き声" | "泣" | "嗚咽" | "すすり泣き" => Some("😭"),

            // === 15. 😱 : 悲鳴、叫び、絶叫 (Scream, shout, shriek) ===
            "😱" | "scream" | "screams" | "screaming" | "shriek" | "shrieks" | "shrieking"
            | "shout" | "shouts" | "shouting" | "yell" | "yells" | "yelling" | "cry out"
            | "holler" | "loud" | "loudly" | "top of lungs" | "style-shout" | "style-shouting"
            | "sfx-scream" | "sfx-screaming"
            | "悲鳴" | "叫び" | "絶叫" | "大声" => Some("😱"),

            // === 16. 😪 : 眠そうに、気だるげに (Sleepily, sluggishly / languidly) ===
            "😪" | "sleepy" | "sleepily" | "drowsy" | "drowsily" | "sluggish" | "sluggishly"
            | "languid" | "languidly" | "listless" | "listlessly" | "lethargic" | "half-asleep"
            | "groggy"
            | "眠い" | "眠そう" | "眠そうに" | "気だるげ" | "気だるげに" | "だるそう" => Some("😪"),

            // === 17. ⏩ : 早口、一気にまくしたてる、急いで (Fast-speaking, rapid-fire, hurriedly) ===
            "⏩" | "fast" | "fast-speaking" | "quickly" | "quick" | "rapid" | "rapidly"
            | "rapid-fire" | "rushed" | "hurried" | "hurriedly" | "hasty" | "hastily"
            | "clipped" | "brisk" | "briskly" | "prosody-fast" | "prosody-speed_fast"
            | "早口" | "まくしたてる" | "一気にまくしたてる" | "急ぎ" | "急いで" => Some("⏩"),

            // === 18. 📞 : 電話越し、スピーカー越しのような音 (Over the phone, through a speaker) ===
            "📞" | "phone" | "telephone" | "over the phone" | "speaker" | "through a speaker"
            | "radio" | "intercom" | "telepathic" | "distant"
            | "電話" | "電話越し" | "スピーカー越し" => Some("📞"),

            // === 19. 🐢 : ゆっくりと (Slowly) ===
            "🐢" | "slow" | "slowly" | "deliberate" | "deliberately" | "measured"
            | "unhurried" | "drawn-out" | "prosody-slow" | "prosody-speed_slow"
            | "ゆっくり" | "ゆっくりと" => Some("🐢"),

            // === 20. 🥤 : 唾を飲み込む音 (Gulp, swallowing sound) ===
            "🥤" | "gulp" | "gulps" | "gulping" | "swallow" | "swallows" | "swallowing"
            | "swallowing sound" | "hard swallow" | "drink" | "drinking"
            | "sfx-gulp" | "sfx-swallow"
            | "飲み込む" | "唾を飲み込む音" | "唾を飲む" | "ゴクリ" | "ごくり" => Some("🥤"),

            // === 21. 🤧 : 咳き込み、鼻をすする、くしゃみ、咳払い (Coughing, sniffling, sneeze, clearing throat) ===
            "🤧" | "clear throat" | "clears throat" | "clearing throat" | "throat-clearing"
            | "ahem" | "cough" | "coughs" | "coughing" | "sniff" | "sniffs" | "sniffling"
            | "sniffle" | "sniffles" | "sneeze" | "sneezes" | "sneezing" | "hiccup"
            | "hiccups" | "choke" | "choking"
            | "sfx-cough" | "sfx-clear-throat" | "sfx-sneeze" | "sfx-sniff"
            | "咳" | "咳き込み" | "咳払い" | "くしゃみ" | "鼻をすする" => Some("🤧"),

            // === 22. 😒 : 舌打ち (Tutting, clicking tongue) ===
            "😒" | "tsk" | "tut" | "tutting" | "click tongue" | "clicks tongue"
            | "clicking tongue" | "tongue click" | "sfx-tsk"
            | "舌打ち" | "チッ" => Some("😒"),

            // === 23. 😰 : 慌てて、動揺、緊張、どもり (Panicked, agitated, nervous, stuttering) ===
            "😰" | "panicked" | "panic" | "panicking" | "agitated" | "agitatedly" | "nervous"
            | "nervously" | "tense" | "tensely" | "stutter" | "stutters" | "stuttering"
            | "stammer" | "stammers" | "stammering" | "flustered" | "frantic" | "frantically"
            | "urgent" | "urgently" | "desperate" | "desperately" | "fear" | "fearful"
            | "fearfully" | "scared" | "terrified" | "frightened" | "afraid" | "horrified"
            | "dread"
            | "emotion-fear" | "emotion-anxious" | "emotion-nervous" | "emotion-panicked"
            | "慌て" | "慌てて" | "動揺" | "緊張" | "どもり" | "焦り" | "取り乱す"
            | "恐れ" | "怖" | "恐怖" => Some("😰"),

            // === 24. 😆 : 喜びながら (Joyfully, happily) ===
            "😆" | "joy" | "joyful" | "joyfully" | "laugh" | "laughs" | "laughter"
            | "laughing" | "cackle" | "cackling" | "guffaw" | "gleeful" | "gleefully"
            | "ecstatic" | "ecstatically" | "elated" | "excited" | "excitedly"
            | "enthusiastic" | "enthusiastically" | "jovial" | "jokingly"
            | "sfx-laughter" | "sfx-laugh" | "emotion-excited" | "emotion-joy" | "emotion-elation"
            | "喜び" | "喜びながら" | "歓喜" | "笑い" | "大笑い" | "笑" | "わくわく" | "興奮" => Some("😆"),

            // === 25. 😠 : 怒り、不満げに、拗ねながら (Angry, displeased, sulking) ===
            "😠" | "angry" | "angrily" | "anger" | "mad" | "furious" | "furiously" | "enraged"
            | "seething" | "hostile" | "hostilely" | "aggressive" | "aggressively" | "fierce"
            | "fiercely" | "menacing" | "menacingly" | "threatening" | "threateningly"
            | "defiant" | "defiantly" | "stern" | "sternly" | "commanding" | "authoritative"
            | "gruff" | "gruffly" | "harsh" | "harshly" | "sharp" | "sharply" | "curt"
            | "curtly" | "bitter" | "bitterly" | "resentful" | "indignant" | "indignantly"
            | "outraged" | "irritated" | "irritatedly" | "frustrated" | "frustratedly"
            | "impatient" | "impatiently" | "displeased" | "dissatisfied" | "dissatisfaction"
            | "dissatisfaction-hnn" | "grumpy"
            | "grumpily" | "sulking" | "sulk" | "sulkily" | "pouting" | "pout" | "pouty"
            | "petulant" | "petulantly" | "disgusted" | "disgust" | "contemptuous"
            | "disdainful" | "repulsed"
            | "emotion-angry" | "emotion-anger" | "emotion-furious" | "emotion-frustrated"
            | "emotion-disgust" | "emotion-bitterness"
            | "怒り" | "怒" | "不満" | "不満げ" | "不満げに" | "拗ね" | "拗ねながら" | "むすっ" => Some("😠"),

            // === 26. 😲 : 驚き、感嘆 (Surprise, awe / exclamation) ===
            "😲" | "surprise" | "surprised" | "surprise-wa" | "surprise-oh" | "surprise-ah"
            | "surprise-yo" | "shock" | "shocked" | "startled" | "astonished"
            | "amazed" | "astounded" | "stunned" | "dumbfounded" | "awed" | "awe" | "in awe"
            | "wonder" | "marveling" | "incredulous" | "incredulously" | "exclamation"
            | "emotion-surprise" | "emotion-surprised" | "emotion-awe"
            | "驚き" | "驚" | "感嘆" | "びっくり" => Some("😲"),

            // === 27. 🥱 : あくび (Yawn) ===
            "🥱" | "yawn" | "yawns" | "yawning" | "stifled yawn" | "sfx-yawn"
            | "あくび" | "欠伸" => Some("🥱"),

            // === 28. 😖 : 苦しげに (Painfully, agonizingly) ===
            "😖" | "pain" | "pained" | "painful" | "painfully" | "agony" | "agonizing"
            | "agonizingly" | "anguish" | "anguished" | "tormented" | "suffering" | "distress"
            | "distressed" | "strained" | "straining" | "wounded" | "injured" | "hoarse"
            | "hoarsely" | "raspy" | "raspily" | "wince" | "winces" | "wincing" | "grimace"
            | "emotion-pain"
            | "苦しい" | "苦しげ" | "苦しげに" | "苦しそう" | "悶え" => Some("😖"),

            // === 29. 😟 : 心配そうに (Anxiously, worriedly) ===
            "😟" | "worried" | "worriedly" | "worry" | "anxious" | "anxiously" | "concerned"
            | "uneasy" | "uneasily" | "troubled" | "apprehensive" | "apprehensively"
            | "foreboding" | "emotion-worried"
            | "心配" | "心配そう" | "心配そうに" | "不安" | "気遣う" => Some("😟"),

            // === 30. 🫣 : 恥ずかしそうに、照れながら (Shyly, bashfully) ===
            "🫣" | "shy" | "shyly" | "bashful" | "bashfully" | "embarrassed" | "embarrassedly"
            | "blushing" | "self-conscious" | "modest" | "demure"
            | "emotion-shy" | "emotion-embarrassed" | "emotion-shame"
            | "恥ずかしい" | "恥ずかしそう" | "恥ずかしそうに" | "照れ" | "照れながら" | "はにかむ" => Some("🫣"),

            // === 31. 🙄 : 呆れたように (Exasperatedly, rolling eyes) ===
            "🙄" | "exasperated" | "exasperatedly" | "annoyed" | "fed up" | "rolling eyes"
            | "eye roll" | "dismissive" | "dismissively" | "unimpressed" | "deadpan"
            | "monotone" | "flat" | "flatly" | "dry" | "dryly" | "cold" | "coldly" | "aloof"
            | "emotion-annoyed"
            | "呆れ" | "呆れたように" | "やれやれ" => Some("🙄"),

            // === 32. 😊 : 楽しげに、嬉しそうに (Cheerfully, gladly) ===
            "😊" | "cheerful" | "cheerfully" | "glad" | "gladly" | "happy" | "happily"
            | "delighted" | "delightedly" | "pleased" | "bright" | "brightly" | "warm smile"
            | "smile" | "smiling" | "hopeful" | "hopefully" | "optimistic" | "proud"
            | "proudly" | "confident" | "confidently" | "assertive" | "assertively"
            | "assisting" | "helpful" | "polite" | "politely" | "regal" | "haughty"
            | "emotion-happy" | "emotion-cheerful" | "emotion-proud" | "emotion-contentment"
            | "楽しげ" | "楽しげに" | "楽しそう" | "嬉しそう" | "嬉しそうに" | "明るく" => Some("😊"),

            // === 33. 👌 : 相槌、頷く音 (Backchanneling, sound of agreement) ===
            "👌" | "agree" | "agreement" | "nod" | "nods" | "nodding" | "backchannel"
            | "backchanneling" | "mm-hmm" | "uh-huh" | "affirmative"
            | "confirmation-en" | "confirmation"
            | "相槌" | "頷く" | "頷く音" | "肯定" => Some("👌"),

            // === 34. 🙏 : 懇願するように (Pleadingly, begging) ===
            "🙏" | "plead" | "pleads" | "pleading" | "pleadingly" | "beg" | "begs" | "begging"
            | "implore" | "imploring" | "imploringly" | "supplicating" | "beseeching"
            | "earnest" | "earnestly"
            | "懇願" | "懇願するように" | "お願い" | "頼み" => Some("🙏"),

            // === 35. 🥴 : 酔っ払って (Drunkenly) ===
            "🥴" | "drunk" | "drunken" | "drunkenly" | "tipsy" | "intoxicated" | "slurred"
            | "slurring" | "woozy" | "dazed"
            | "酔っ払い" | "酔っ払って" | "泥酔" | "ほろ酔い" => Some("🥴"),

            // === 36. 🎵 : 鼻歌 (Humming) ===
            "🎵" | "hum" | "hums" | "humming" | "sing" | "sings" | "singing" | "song"
            | "melodic" | "melodically" | "lullaby" | "chant" | "chanting"
            | "style-singing" | "style-hum" | "sfx-humming"
            | "鼻歌" | "歌" => Some("🎵"),

            // === 37. 🤐 : 口を塞がれて (Muffled (mouth covered)) ===
            "🤐" | "muffled" | "gagged" | "mouth covered" | "stifled" | "muted"
            | "口を塞がれて" | "こもった声" | "くぐもった声" => Some("🤐"),

            // === 38. 😌 : 安堵、満足げに (Relieved, contentedly) ===
            "😌" | "relieved" | "relief" | "content" | "contented" | "contentedly"
            | "satisfied" | "satisfiedly" | "peaceful" | "peacefully" | "serene" | "serenely"
            | "composed" | "serious" | "seriously" | "solemn" | "solemnly"
            | "grave" | "gravely" | "grim" | "grimly" | "determined" | "resolute"
            | "narration" | "narrator"
            | "emotion-relieved" | "emotion-relief" | "emotion-serious" | "style-narration"
            | "安堵" | "ほっとした" | "満足" | "満足げ" | "満足げに" => Some("😌"),

            // === 39. 🤔 : 疑問の声 (Questioning voice, wondering) ===
            "🤔" | "questioning" | "questioningly" | "questioning voice" | "question-oh"
            | "wondering" | "wonderingly" | "curious" | "curiously" | "puzzled"
            | "confused" | "confusedly" | "perplexed" | "baffled" | "skeptical"
            | "skeptically" | "doubtful" | "doubtfully" | "suspicious" | "suspiciously"
            | "thoughtful" | "thoughtfully" | "pensive" | "pensively" | "contemplative"
            | "pondering" | "musings"
            | "emotion-curious" | "emotion-confused" | "emotion-confusion"
            | "疑問" | "疑問の声" | "思案" | "不思議そう" => Some("🤔"),

            // Unknown tags: safely remove so they are never spoken aloud
            _ => None,
        }
    }

    /// Resolves a raw tag string into an emoji sequence, supporting intensifier prefixes
    /// (e.g. `"very angry"` -> `"😠😠"`, `"deep sigh"` -> `"😮‍💨😮‍💨"`) and softener prefixes
    /// (e.g. `"slightly shy"` -> `"🫣"`).
    pub fn resolve_tag_tokens(tag: &str) -> Option<String> {
        let normalized = tag.trim().to_lowercase();

        // Direct exact match
        if let Some(emoji) = Self::tag_to_emoji(&normalized) {
            return Some(emoji.to_string());
        }

        // Intensifier prefix -> repeat emoji twice for emphasis
        for prefix in INTENSIFIER_PREFIXES {
            if let Some(rest) = normalized.strip_prefix(prefix) {
                if let Some(emoji) = Self::tag_to_emoji(rest) {
                    return Some(format!("{emoji}{emoji}"));
                }
            }
        }

        // Softener prefix -> single emoji
        for prefix in SOFTENER_PREFIXES {
            if let Some(rest) = normalized.strip_prefix(prefix) {
                if let Some(emoji) = Self::tag_to_emoji(rest) {
                    return Some(emoji.to_string());
                }
            }
        }

        None
    }

    fn classify_raw_layer(raw_tag: &str) -> Option<TagLayer> {
        let base = EmotionParser::strip_modifier_prefix(raw_tag.trim());
        EmotionParser::classify_tag(base).map(|p| p.layer)
    }

    fn convert_single_segment(
        segment: &str,
        inherited_carry_emoji: &mut Option<String>,
        inherited_mood_emoji: &mut Option<String>,
        calm_reset_active: &mut bool,
        detected_tags: &mut Vec<String>,
    ) -> String {
        let input_no_se = SE_WRAPPER_REGEX.replace_all(segment, " ");

        // Pre-scan enclosed tags in this segment to see whether an explicit Carry or Reset tag
        // appears here (so `[mood:<tag>]` won't emit a duplicate conflicting emoji).
        let mut segment_has_explicit_carry_or_reset = false;
        for caps in FULL_TAG_REGEX.captures_iter(&input_no_se) {
            let raw_tag = caps
                .name("tag")
                .or_else(|| caps.name("ztag"))
                .or_else(|| caps.name("ptag"))
                .or_else(|| caps.name("fptag"))
                .or_else(|| caps.name("atag"))
                .map(|m| m.as_str().trim().to_lowercase())
                .unwrap_or_default();
            if raw_tag.is_empty() || raw_tag.starts_with("mood:") {
                continue;
            }
            if matches!(
                Self::classify_raw_layer(&raw_tag),
                Some(TagLayer::Carry | TagLayer::Reset)
            ) {
                segment_has_explicit_carry_or_reset = true;
                break;
            }
        }

        let mut local_mood_emitted = false;

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

            if let Some(mood_inner) = raw_tag.strip_prefix("mood:") {
                if let Some(emoji) = Self::resolve_tag_tokens(mood_inner) {
                    *inherited_mood_emoji = Some(emoji.clone());
                    if !segment_has_explicit_carry_or_reset
                        && inherited_carry_emoji.is_none()
                        && !*calm_reset_active
                    {
                        local_mood_emitted = true;
                        return format!(" {emoji} ");
                    }
                }
                return " ".to_string();
            }

            match Self::classify_raw_layer(&raw_tag) {
                Some(TagLayer::Reset) => {
                    *calm_reset_active = true;
                    *inherited_carry_emoji = None;
                    return " ".to_string();
                }
                Some(TagLayer::Carry) => {
                    *calm_reset_active = false;
                    if let Some(emoji) = Self::resolve_tag_tokens(&raw_tag) {
                        *inherited_carry_emoji = Some(emoji.clone());
                        return format!(" {emoji} ");
                    }
                }
                _ => {}
            }

            match Self::resolve_tag_tokens(&raw_tag) {
                Some(emoji) => format!(" {emoji} "),
                None => " ".to_string(),
            }
        });

        // 2. Rescue tags missing opening bracket (e.g. "chuckle] Hello")
        let text_after_missing_open =
            MISSING_OPEN_BRACKET_REGEX.replace_all(&text_after_full, |caps: &regex::Captures| {
                let prefix = caps.name("prefix").map(|m| m.as_str()).unwrap_or("");
                let raw_tag = caps
                    .name("otag")
                    .map(|m| m.as_str().trim().to_lowercase())
                    .unwrap_or_default();

                if !raw_tag.is_empty() {
                    detected_tags.push(raw_tag.clone());
                }

                if matches!(Self::classify_raw_layer(&raw_tag), Some(TagLayer::Carry)) {
                    *calm_reset_active = false;
                    if let Some(emoji) = Self::resolve_tag_tokens(&raw_tag) {
                        *inherited_carry_emoji = Some(emoji.clone());
                        segment_has_explicit_carry_or_reset = true;
                        return format!("{prefix} {emoji} ");
                    }
                }

                match Self::resolve_tag_tokens(&raw_tag) {
                    Some(emoji) => format!("{prefix} {emoji} "),
                    None => format!("{prefix} "),
                }
            });

        // 3. Rescue tags missing closing bracket (e.g. "[whisper Hello")
        let text_after_missing_close = MISSING_CLOSE_BRACKET_REGEX.replace_all(
            &text_after_missing_open,
            |caps: &regex::Captures| {
                let suffix = caps.name("suffix").map(|m| m.as_str()).unwrap_or("");
                let raw_tag = caps
                    .name("itag")
                    .map(|m| m.as_str().trim().to_lowercase())
                    .unwrap_or_default();

                if !raw_tag.is_empty() {
                    detected_tags.push(raw_tag.clone());
                }

                if matches!(Self::classify_raw_layer(&raw_tag), Some(TagLayer::Carry)) {
                    *calm_reset_active = false;
                    if let Some(emoji) = Self::resolve_tag_tokens(&raw_tag) {
                        *inherited_carry_emoji = Some(emoji.clone());
                        segment_has_explicit_carry_or_reset = true;
                        return format!(" {emoji} {suffix}");
                    }
                }

                match Self::resolve_tag_tokens(&raw_tag) {
                    Some(emoji) => format!(" {emoji} {suffix}"),
                    None => format!(" {suffix}"),
                }
            },
        );

        let collapsed = MULTI_SPACE_REGEX
            .replace_all(&text_after_missing_close, " ")
            .to_string();
        let trimmed = collapsed.trim();
        let sanitized = ORPHAN_BRACKETS_REGEX.replace_all(trimmed, "").to_string();
        let clean_seg = sanitized.trim().to_string();

        // If this speakable sentence had no explicit Carry/Reset tag of its own, prepend the active
        // Carry emoji (or active BaseMood emoji if no Carry was set and not reset).
        if !segment_has_explicit_carry_or_reset && SPEAKABLE_CHAR_REGEX.is_match(&clean_seg) {
            if let Some(ref carry_emoji) = *inherited_carry_emoji {
                return format!("{carry_emoji} {clean_seg}");
            } else if !local_mood_emitted && !*calm_reset_active {
                if let Some(ref mood_emoji) = *inherited_mood_emoji {
                    return format!("{mood_emoji} {clean_seg}");
                }
            }
        }

        clean_seg
    }

    /// Converts raw text containing audio/emotion tags into Irodori emoji-embedded text,
    /// propagating Carry Tags across sentences and honoring `[calm]` / `[reset]` and `[mood:<tag>]`.
    pub fn convert(input: &str) -> IrodoriMappedText {
        let input_no_se = SE_WRAPPER_REGEX.replace_all(input, " ").to_string();
        let raw_sentences = EmotionParser::split_raw_sentences(&input_no_se);

        let mut detected_tags = Vec::new();
        let mut carry_emoji: Option<String> = None;
        let mut mood_emoji: Option<String> = None;
        let mut calm_reset_active = false;

        if raw_sentences.len() <= 1 {
            let prompt_text = Self::convert_single_segment(
                &input_no_se,
                &mut carry_emoji,
                &mut mood_emoji,
                &mut calm_reset_active,
                &mut detected_tags,
            );
            return IrodoriMappedText {
                prompt_text,
                detected_tags,
            };
        }

        // Merge trailing tag-only fragments (e.g. "...they'll hear us. [sigh]") with preceding sentence
        // and leading tag-only fragments with the next sentence, matching EmotionParser::parse_chunks.
        let mut merged_sentences: Vec<String> = Vec::new();
        let mut pending_prefix = String::new();

        for seg in raw_sentences {
            let stripped = FULL_TAG_REGEX.replace_all(&seg, " ");
            if SPEAKABLE_CHAR_REGEX.is_match(&stripped) {
                if pending_prefix.is_empty() {
                    merged_sentences.push(seg);
                } else {
                    merged_sentences.push(format!("{pending_prefix} {seg}"));
                    pending_prefix.clear();
                }
            } else if merged_sentences.is_empty() {
                if pending_prefix.is_empty() {
                    pending_prefix = seg;
                } else {
                    pending_prefix.push(' ');
                    pending_prefix.push_str(&seg);
                }
            } else if let Some(last) = merged_sentences.last_mut() {
                last.push(' ');
                last.push_str(&seg);
            }
        }
        if !pending_prefix.is_empty() {
            if let Some(last) = merged_sentences.last_mut() {
                last.push(' ');
                last.push_str(&pending_prefix);
            } else {
                merged_sentences.push(pending_prefix);
            }
        }

        let mut rendered_parts = Vec::with_capacity(merged_sentences.len());
        for seg in merged_sentences {
            let part = Self::convert_single_segment(
                &seg,
                &mut carry_emoji,
                &mut mood_emoji,
                &mut calm_reset_active,
                &mut detected_tags,
            );
            if !part.is_empty() {
                rendered_parts.push(part);
            }
        }

        let prompt_text = MULTI_SPACE_REGEX
            .replace_all(&rendered_parts.join(" "), " ")
            .trim()
            .to_string();

        IrodoriMappedText {
            prompt_text,
            detected_tags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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
    fn test_irodori_all_39_official_emojis_coverage() {
        // Every single one of the 39 emojis from Aratako/Irodori-TTS-500M EMOJI_ANNOTATIONS.md
        let cases: &[(&str, &str)] = &[
            ("whisper", "👂"),        // 1. 囁き、耳元の音
            ("sigh", "😮‍💨"),           // 2. 吐息、溜息、寝息
            ("pause", "⏸️"),          // 3. 間、沈黙
            ("chuckle", "🤭"),        // 4. 笑い（くすくす、含み笑いなど）
            ("moan", "🥵"),           // 5. 喘ぎ、うめき声、唸り声
            ("echo", "📢"),           // 6. エコー、リバーブ
            ("coaxing", "😏"),        // 7. からかうように、甘えるように
            ("trembling", "🥺"),      // 8. 声を震わせながら、自信のなさげに
            ("panting", "🌬️"),        // 9. 息切れ、荒い息遣い、呼吸音
            ("gasp", "😮"),           // 10. 息をのむ
            ("lick", "👅"),           // 11. 舐める音、咀嚼音、水音
            ("lip smack", "💋"),      // 12. リップノイズ
            ("gently", "🫶"),         // 13. 優しく
            ("sobbing", "😭"),        // 14. 嗚咽、泣き声、悲しみ
            ("screaming", "😱"),      // 15. 悲鳴、叫び、絶叫
            ("sleepily", "😪"),       // 16. 眠そうに、気だるげに
            ("fast", "⏩"),           // 17. 早口、一気にまくしたてる、急いで
            ("phone", "📞"),          // 18. 電話越し、スピーカー越しのような音
            ("slowly", "🐢"),         // 19. ゆっくりと
            ("gulp", "🥤"),           // 20. 唾を飲み込む音
            ("clear throat", "🤧"),   // 21. 咳き込み、鼻をすする、くしゃみ、咳払い
            ("tsk", "😒"),            // 22. 舌打ち
            ("stuttering", "😰"),     // 23. 慌てて、動揺、緊張、どもり
            ("joyfully", "😆"),       // 24. 喜びながら
            ("sulking", "😠"),        // 25. 怒り、不満げに、拗ねながら
            ("surprised", "😲"),      // 26. 驚き、感嘆
            ("yawn", "🥱"),           // 27. あくび
            ("pained", "😖"),         // 28. 苦しげに
            ("worried", "😟"),        // 29. 心配そうに
            ("shyly", "🫣"),          // 30. 恥ずかしそうに、照れながら
            ("exasperated", "🙄"),    // 31. 呆れたように
            ("cheerfully", "😊"),     // 32. 楽しげに、嬉しそうに
            ("nod", "👌"),            // 33. 相槌、頷く音
            ("pleading", "🙏"),       // 34. 懇願するように
            ("drunkenly", "🥴"),      // 35. 酔っ払って
            ("humming", "🎵"),        // 36. 鼻歌
            ("muffled", "🤐"),        // 37. 口を塞がれて
            ("relieved", "😌"),       // 38. 安堵、満足げに
            ("questioning", "🤔"),    // 39. 疑問の声
        ];

        assert_eq!(cases.len(), 39);
        let mut unique_emojis = HashSet::new();
        for (tag, expected_emoji) in cases {
            let actual = IrodoriMapper::tag_to_emoji(tag);
            assert_eq!(
                actual,
                Some(*expected_emoji),
                "Tag '{tag}' should map to official emoji '{expected_emoji}'"
            );
            unique_emojis.insert(*expected_emoji);
        }
        assert_eq!(
            unique_emojis.len(),
            39,
            "All 39 distinct official Irodori emojis must be uniquely covered"
        );
    }

    #[test]
    fn test_irodori_intensifier_and_softener_prefixes() {
        let input = "[very angry] もう知らない！ [deep sigh] [slightly shy] …嘘だよ。";
        let mapped = IrodoriMapper::convert(input);
        assert!(mapped.prompt_text.contains("😠😠"));
        assert!(mapped.prompt_text.contains("😮‍💨😮‍💨"));
        assert!(mapped.prompt_text.contains("🫣"));
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

    #[test]
    fn test_irodori_multi_sentence_carry_mood_and_calm_reset() {
        let input = "[mood:gently] お疲れ様。 [whisper] 内緒だよ。 誰にも言わないでね。 [calm] さて本題だよ。";
        let mapped = IrodoriMapper::convert(input);
        assert_eq!(
            mapped.prompt_text,
            "🫶 お疲れ様。 👂 内緒だよ。 👂 誰にも言わないでね。 さて本題だよ。"
        );

        // Hyphenated nina_agent reaction aliases
        let reactions = IrodoriMapper::convert("[surprise-wa] えっ！ [dissatisfaction-hnn] むー！ [confirmation-en] うん！ [question-oh] なに？");
        assert!(reactions.prompt_text.contains("😲"));
        assert!(reactions.prompt_text.contains("😠"));
        assert!(reactions.prompt_text.contains("👌"));
        assert!(reactions.prompt_text.contains("🤔"));

        // All kitt_agent non-neutral moods map cleanly to official Irodori emojis
        for (kitt_mood, expected) in [
            ("assisting", "😊"),
            ("lovely", "🫶"),
            ("sassy", "😏"),
            ("assertive", "😊"),
            ("sardonic", "😏"),
            ("smirking", "😏"),
        ] {
            assert_eq!(IrodoriMapper::tag_to_emoji(kitt_mood), Some(expected));
        }
    }
}
