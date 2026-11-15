use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "openai-tts-bridge",
    about = "High-performance OpenAI-compatible TTS bridge supporting Zonos v0.1, Zonos 2, and Irodori-TTS",
    version
)]
pub struct Config {
    /// Default TTS engine to use ("zonos", "zonos2", or "irodori")
    #[arg(long, env = "TTS_ENGINE", default_value = "zonos2")]
    pub tts_engine: String,

    /// Host to bind the bridge server to
    #[arg(long, env = "HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on
    #[arg(short, long, env = "PORT", default_value_t = 8080)]
    pub port: u16,

    /// Base URL of the Zonos v0.1 Gradio server
    #[arg(long, env = "ZONOS_V1_URL", default_value = "http://127.0.0.1:7860")]
    pub zonos_v1_url: String,

    /// Base URL of the Zonos 2 FastAPI server
    #[arg(long, env = "ZONOS_URL", default_value = "http://127.0.0.1:1919")]
    pub zonos_url: String,

    /// Base URL of the Irodori-TTS-Server
    #[arg(long, env = "IRODORI_URL", default_value = "http://127.0.0.1:8088")]
    pub irodori_url: String,

    /// Default model name for Irodori-TTS
    #[arg(long, env = "IRODORI_MODEL", default_value = "irodori-tts")]
    pub irodori_model: String,

    /// Default speaker embedding voice name
    #[arg(long, env = "DEFAULT_VOICE", default_value = "default")]
    pub default_voice: String,

    /// Default model name returned in model list
    #[arg(long, env = "DEFAULT_MODEL", default_value = "zonos2")]
    pub default_model: String,

    /// Default language code ("ja", "en-us", etc.)
    #[arg(long, env = "DEFAULT_LANGUAGE", default_value = "ja")]
    pub default_language: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Maximum request body size in megabytes (for large base64 speaker audio)
    #[arg(long, env = "MAX_BODY_SIZE_MB", default_value_t = 100)]
    pub max_body_size_mb: usize,

    /// Default speech speed factor (1.15 is natural tempo for Japanese in Zonos 2)
    #[arg(long, env = "DEFAULT_SPEED", default_value_t = 1.15)]
    pub default_speed: f32,

    /// Path to persistent TOML runtime configuration file (e.g. "./config.toml")
    #[arg(long, env = "CONFIG_FILE")]
    pub config_file: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tts_engine: "zonos2".to_string(),
            host: "0.0.0.0".to_string(),
            port: 8080,
            zonos_v1_url: "http://127.0.0.1:7860".to_string(),
            zonos_url: "http://127.0.0.1:1919".to_string(),
            irodori_url: "http://127.0.0.1:8088".to_string(),
            irodori_model: "irodori-tts".to_string(),
            default_voice: "default".to_string(),
            default_model: "zonos2".to_string(),
            default_language: "ja".to_string(),
            log_level: "info".to_string(),
            max_body_size_mb: 100,
            default_speed: 1.15,
            config_file: None,
        }
    }
}
