use crate::config::Config;
use crate::error::AppError;
use crate::normalizer::{CompiledDictionary, DictionaryEntry, EngineNormFlags};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_engine_name() -> String {
    "zonos2".to_string()
}

fn default_voice_name() -> String {
    "nina".to_string()
}

fn default_language_code() -> String {
    "ja".to_string()
}

fn default_zonos_v1_url() -> String {
    "http://127.0.0.1:7860".to_string()
}

fn default_zonos2_url() -> String {
    "http://127.0.0.1:1919".to_string()
}

fn default_irodori_url() -> String {
    "http://127.0.0.1:8088".to_string()
}

fn default_irodori_model() -> String {
    "irodori-tts".to_string()
}

fn default_speaking_rate() -> f32 {
    15.0
}

fn default_pitch_std() -> f32 {
    45.0
}

fn default_chunk_silence_ms() -> u32 {
    100
}

fn default_zonos2_speed() -> f32 {
    1.15
}

fn default_irodori_speed() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeneralSettings {
    #[serde(default = "default_engine_name")]
    pub default_engine: String,
    #[serde(default = "default_voice_name")]
    pub default_voice: String,
    #[serde(default = "default_language_code")]
    pub default_language: String,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            default_engine: default_engine_name(),
            default_voice: default_voice_name(),
            default_language: default_language_code(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZonosV1Settings {
    #[serde(default = "default_zonos_v1_url")]
    pub url: String,
    #[serde(default = "default_speaking_rate")]
    pub default_speaking_rate: f32,
    #[serde(default = "default_pitch_std")]
    pub default_pitch_std: f32,
    #[serde(default = "default_chunk_silence_ms")]
    pub chunk_silence_ms: u32,
    #[serde(default = "default_true")]
    pub use_dictionary: bool,
    #[serde(default = "default_true")]
    pub fix_wave_dash: bool,
    #[serde(default = "default_true")]
    pub normalize_punctuation: bool,
    #[serde(default = "default_true")]
    pub convert_units: bool,
}

impl Default for ZonosV1Settings {
    fn default() -> Self {
        Self {
            url: default_zonos_v1_url(),
            default_speaking_rate: default_speaking_rate(),
            default_pitch_std: default_pitch_std(),
            chunk_silence_ms: default_chunk_silence_ms(),
            use_dictionary: true,
            fix_wave_dash: true,
            normalize_punctuation: true,
            convert_units: true,
        }
    }
}

impl ZonosV1Settings {
    pub fn norm_flags(&self) -> EngineNormFlags {
        EngineNormFlags {
            use_dictionary: self.use_dictionary,
            fix_wave_dash: self.fix_wave_dash,
            normalize_punctuation: self.normalize_punctuation,
            convert_units: self.convert_units,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Zonos2Settings {
    #[serde(default = "default_zonos2_url")]
    pub url: String,
    #[serde(default = "default_zonos2_speed")]
    pub default_speed: f32,
    #[serde(default = "default_true")]
    pub use_dictionary: bool,
    #[serde(default = "default_true")]
    pub fix_wave_dash: bool,
    #[serde(default = "default_true")]
    pub normalize_punctuation: bool,
    #[serde(default = "default_true")]
    pub convert_units: bool,
}

impl Default for Zonos2Settings {
    fn default() -> Self {
        Self {
            url: default_zonos2_url(),
            default_speed: default_zonos2_speed(),
            use_dictionary: true,
            fix_wave_dash: true,
            normalize_punctuation: true,
            convert_units: true,
        }
    }
}

impl Zonos2Settings {
    pub fn norm_flags(&self) -> EngineNormFlags {
        EngineNormFlags {
            use_dictionary: self.use_dictionary,
            fix_wave_dash: self.fix_wave_dash,
            normalize_punctuation: self.normalize_punctuation,
            convert_units: self.convert_units,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IrodoriSettings {
    #[serde(default = "default_irodori_url")]
    pub url: String,
    #[serde(default = "default_irodori_model")]
    pub model: String,
    #[serde(default = "default_irodori_speed")]
    pub default_speed: f32,
    #[serde(default = "default_true")]
    pub use_dictionary: bool,
    #[serde(default = "default_false")]
    pub fix_wave_dash: bool,
    #[serde(default = "default_false")]
    pub normalize_punctuation: bool,
    #[serde(default = "default_false")]
    pub convert_units: bool,
}

impl Default for IrodoriSettings {
    fn default() -> Self {
        Self {
            url: default_irodori_url(),
            model: default_irodori_model(),
            default_speed: default_irodori_speed(),
            use_dictionary: true,
            fix_wave_dash: false,
            normalize_punctuation: false,
            convert_units: false,
        }
    }
}

impl IrodoriSettings {
    pub fn norm_flags(&self) -> EngineNormFlags {
        EngineNormFlags {
            use_dictionary: self.use_dictionary,
            fix_wave_dash: self.fix_wave_dash,
            normalize_punctuation: self.normalize_punctuation,
            convert_units: self.convert_units,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EnginesSettings {
    #[serde(default)]
    pub zonos: ZonosV1Settings,
    #[serde(default)]
    pub zonos2: Zonos2Settings,
    #[serde(default)]
    pub irodori: IrodoriSettings,
}

pub fn default_dictionary_entries() -> Vec<DictionaryEntry> {
    let pairs = [
        ("東白川村", "ひがししらかわむら", true),
        ("東白川", "ひがししらかわ", true),
        ("mBot2", "エムボットツー", false),
        ("mBot", "エムボット", false),
        ("CyberPi", "サイバーパイ", false),
        ("Makeblock", "メイクブロック", false),
        ("mBuild", "エムビルド", false),
        ("K.I.T.T.", "キット", false),
        ("KITT", "キット", true),
        ("Kitt", "キット", true),
        ("Knight Rider", "ナイトライダー", false),
        ("LED", "エルイーディー", true),
        ("RGB", "アールジービー", true),
        ("RPM", "アールピーエム", true),
        ("Wi-Fi", "ワイファイ", false),
        ("WiFi", "ワイファイ", false),
        ("AI", "エーアイ", true),
    ];

    pairs
        .into_iter()
        .map(|(surface, reading, case_sensitive)| DictionaryEntry {
            surface: surface.to_string(),
            reading: reading.to_string(),
            case_sensitive,
            enabled: true,
        })
        .collect()
}

pub fn normalize_engine_key(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "zonos" | "zonos-v0.1" | "zonos_v1" | "zonos1" | "zonos-1" => Some("zonos"),
        "zonos2" | "zonos-2" | "zonos_2" => Some("zonos2"),
        "irodori" | "irodori-tts" => Some("irodori"),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeSettings {
    #[serde(default)]
    pub general: GeneralSettings,
    #[serde(default)]
    pub engines: EnginesSettings,
    #[serde(default = "default_dictionary_entries")]
    pub dictionary: Vec<DictionaryEntry>,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            general: GeneralSettings::default(),
            engines: EnginesSettings::default(),
            dictionary: default_dictionary_entries(),
        }
    }
}

impl RuntimeSettings {
    /// Constructs initial RuntimeSettings seeded from CLI / environment `Config`.
    pub fn from_bootstrap(config: &Config) -> Self {
        let default_engine =
            normalize_engine_key(&config.tts_engine).unwrap_or("zonos2").to_string();

        Self {
            general: GeneralSettings {
                default_engine,
                default_voice: config.default_voice.trim().to_string(),
                default_language: config.default_language.trim().to_string(),
            },
            engines: EnginesSettings {
                zonos: ZonosV1Settings {
                    url: config.zonos_v1_url.trim_end_matches('/').to_string(),
                    ..ZonosV1Settings::default()
                },
                zonos2: Zonos2Settings {
                    url: config.zonos_url.trim_end_matches('/').to_string(),
                    default_speed: config.default_speed,
                    ..Zonos2Settings::default()
                },
                irodori: IrodoriSettings {
                    url: config.irodori_url.trim_end_matches('/').to_string(),
                    model: config.irodori_model.trim().to_string(),
                    ..IrodoriSettings::default()
                },
            },
            dictionary: default_dictionary_entries(),
        }
    }

    pub fn validate_and_normalize(&mut self) -> Result<(), AppError> {
        let Some(canon_engine) = normalize_engine_key(&self.general.default_engine) else {
            return Err(AppError::BadRequest(format!(
                "Invalid default_engine '{}'. Must be 'zonos', 'zonos2', or 'irodori'.",
                self.general.default_engine
            )));
        };
        self.general.default_engine = canon_engine.to_string();

        self.general.default_voice = self.general.default_voice.trim().to_string();
        if self.general.default_voice.is_empty() {
            self.general.default_voice = "default".to_string();
        }

        self.general.default_language = self.general.default_language.trim().to_string();
        if self.general.default_language.is_empty() {
            self.general.default_language = "ja".to_string();
        }

        for (label, url) in [
            ("engines.zonos.url", &mut self.engines.zonos.url),
            ("engines.zonos2.url", &mut self.engines.zonos2.url),
            ("engines.irodori.url", &mut self.engines.irodori.url),
        ] {
            *url = url.trim().trim_end_matches('/').to_string();
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err(AppError::BadRequest(format!(
                    "{label} must start with http:// or https:// (got '{url}')"
                )));
            }
        }

        self.engines.zonos.default_speaking_rate =
            self.engines.zonos.default_speaking_rate.clamp(5.0, 30.0);
        self.engines.zonos.default_pitch_std =
            self.engines.zonos.default_pitch_std.clamp(5.0, 200.0);
        self.engines.zonos.chunk_silence_ms = self.engines.zonos.chunk_silence_ms.min(2000);

        self.engines.zonos2.default_speed = self.engines.zonos2.default_speed.clamp(0.25, 4.0);
        self.engines.irodori.default_speed = self.engines.irodori.default_speed.clamp(0.25, 4.0);

        self.engines.irodori.model = self.engines.irodori.model.trim().to_string();
        if self.engines.irodori.model.is_empty() {
            self.engines.irodori.model = "irodori-tts".to_string();
        }

        self.dictionary.retain(|entry| !entry.surface.trim().is_empty());
        for entry in &mut self.dictionary {
            entry.surface = entry.surface.trim().to_string();
        }

        Ok(())
    }

    pub fn load_from_toml(path: &Path) -> Result<Self, AppError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            AppError::Internal(anyhow::anyhow!(
                "Failed to read config TOML '{}': {e}",
                path.display()
            ))
        })?;
        let mut settings: Self = toml::from_str(&content).map_err(|e| {
            AppError::BadRequest(format!(
                "Failed to parse config TOML '{}': {e}",
                path.display()
            ))
        })?;
        settings.validate_and_normalize()?;
        Ok(settings)
    }

    /// Saves settings to TOML atomically via temporary file + fsync + rename.
    pub fn save_atomic(&self, path: &Path) -> Result<(), AppError> {
        let serialized = toml::to_string_pretty(self).map_err(|e| {
            AppError::Internal(anyhow::anyhow!("Failed to serialize TOML settings: {e}"))
        })?;

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    AppError::Internal(anyhow::anyhow!(
                        "Failed to create config directory '{}': {e}",
                        parent.display()
                    ))
                })?;
            }
        }

        let tmp_name = format!(
            "{}.tmp.{}",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("config.toml"),
            std::process::id()
        );
        let tmp_path = path.with_file_name(tmp_name);

        {
            let mut file = File::create(&tmp_path).map_err(|e| {
                AppError::Internal(anyhow::anyhow!(
                    "Failed to create temporary config file '{}': {e}",
                    tmp_path.display()
                ))
            })?;
            file.write_all(serialized.as_bytes()).map_err(|e| {
                AppError::Internal(anyhow::anyhow!(
                    "Failed to write temporary config file '{}': {e}",
                    tmp_path.display()
                ))
            })?;
            file.sync_all().map_err(|e| {
                AppError::Internal(anyhow::anyhow!(
                    "Failed to fsync temporary config file '{}': {e}",
                    tmp_path.display()
                ))
            })?;
        }

        std::fs::rename(&tmp_path, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_path);
            AppError::Internal(anyhow::anyhow!(
                "Failed to atomically rename '{}' to '{}': {e}",
                tmp_path.display(),
                path.display()
            ))
        })?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CompiledSettings {
    pub raw: RuntimeSettings,
    pub dictionary: CompiledDictionary,
}

impl CompiledSettings {
    pub fn new(mut raw: RuntimeSettings) -> Result<Self, AppError> {
        raw.validate_and_normalize()?;
        let dictionary = CompiledDictionary::compile(&raw.dictionary);
        Ok(Self { raw, dictionary })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_atomic_save_and_load_roundtrip() {
        let tmp_dir = std::env::temp_dir();
        let test_path = tmp_dir.join(format!("openai_tts_bridge_test_{}.toml", std::process::id()));
        let _ = std::fs::remove_file(&test_path);

        let mut settings = RuntimeSettings::default();
        settings.general.default_engine = "irodori".to_string();
        settings.engines.zonos2.default_speed = 1.25;
        settings.dictionary.push(DictionaryEntry {
            surface: "ウィンターホールド".to_string(),
            reading: "うぃんたーほーるど".to_string(),
            case_sensitive: false,
            enabled: true,
        });

        settings.save_atomic(&test_path).expect("save_atomic failed");
        assert!(test_path.exists());

        let loaded = RuntimeSettings::load_from_toml(&test_path).expect("load_from_toml failed");
        assert_eq!(loaded.general.default_engine, "irodori");
        assert_eq!(loaded.engines.zonos2.default_speed, 1.25);
        assert!(loaded
            .dictionary
            .iter()
            .any(|d| d.surface == "ウィンターホールド"));

        let _ = std::fs::remove_file(&test_path);
    }
}
