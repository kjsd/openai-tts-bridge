use crate::config::Config;
use crate::engine::{IrodoriEngine, TtsEngine, ZonosEngine, ZonosV1Engine};
use crate::error::AppError;
use crate::settings::{CompiledSettings, RuntimeSettings};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tracing::{info, warn};

pub struct AppState {
    pub config: Config,
    /// Optional persistent TOML file path (`None` in unit/integration tests unless explicitly configured)
    pub config_file: Option<PathBuf>,
    /// Live runtime settings + compiled dictionary snapshot
    pub settings: Arc<RwLock<Arc<CompiledSettings>>>,
    /// Global GPU mutex shared across all local TTS backends
    pub gpu_lock: Arc<Mutex<()>>,
    /// Concrete Zonos v0.1 engine handle (for cache invalidation on URL change)
    pub zonos_v1_concrete: Arc<ZonosV1Engine>,
    /// Concrete Irodori engine handle (for cache invalidation on URL change)
    pub irodori_concrete: Arc<IrodoriEngine>,
    /// Zonos v0.1 Gradio backend engine instance
    pub zonos_v1_engine: Arc<dyn TtsEngine>,
    /// Zonos 2 FastAPI backend engine instance
    pub zonos_engine: Arc<dyn TtsEngine>,
    /// Irodori-TTS backend engine instance
    pub irodori_engine: Arc<dyn TtsEngine>,
    /// Optional injected default engine override (used by `AppState::with_engine`)
    custom_default_engine: Option<Arc<dyn TtsEngine>>,
    /// Shared in-memory store of zero-shot reference WAVs uploaded via `POST /v1/audio/voices`
    pub enrolled_voices: Mutex<HashMap<String, Vec<u8>>>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        Self::build_internal(config, None)
    }

    pub fn with_engine(config: Config, engine: Arc<dyn TtsEngine>) -> Self {
        Self::build_internal(config, Some(engine))
    }

    fn build_internal(config: Config, injected: Option<Arc<dyn TtsEngine>>) -> Self {
        let config_file = config
            .config_file
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        let runtime_settings = if let Some(ref path) = config_file {
            if path.exists() {
                match RuntimeSettings::load_from_toml(path) {
                    Ok(loaded) => {
                        info!(path = %path.display(), "Loaded runtime settings from TOML");
                        loaded
                    }
                    Err(e) => {
                        warn!(
                            path = %path.display(),
                            error = %e,
                            "Failed to load TOML settings; falling back to bootstrap CLI/env config"
                        );
                        RuntimeSettings::from_bootstrap(&config)
                    }
                }
            } else {
                RuntimeSettings::from_bootstrap(&config)
            }
        } else {
            RuntimeSettings::from_bootstrap(&config)
        };

        let compiled = CompiledSettings::new(runtime_settings)
            .expect("Failed to compile initial RuntimeSettings");
        let settings = Arc::new(RwLock::new(Arc::new(compiled)));
        let gpu_lock = Arc::new(Mutex::new(()));

        let zonos_v1_concrete =
            Arc::new(ZonosV1Engine::with_shared_state(settings.clone(), gpu_lock.clone()));
        let zonos2_concrete =
            Arc::new(ZonosEngine::with_shared_state(settings.clone(), gpu_lock.clone()));
        let irodori_concrete =
            Arc::new(IrodoriEngine::with_shared_state(settings.clone(), gpu_lock.clone()));

        let mut zonos_v1_engine: Arc<dyn TtsEngine> = zonos_v1_concrete.clone();
        let mut zonos_engine: Arc<dyn TtsEngine> = zonos2_concrete;
        let mut irodori_engine: Arc<dyn TtsEngine> = irodori_concrete.clone();
        let mut custom_default_engine: Option<Arc<dyn TtsEngine>> = None;

        if let Some(eng) = injected {
            match eng.name() {
                "zonos" => zonos_v1_engine = eng,
                "zonos2" => zonos_engine = eng,
                "irodori" => irodori_engine = eng,
                _ => custom_default_engine = Some(eng),
            }
        }

        Self {
            config,
            config_file,
            settings,
            gpu_lock,
            zonos_v1_concrete,
            irodori_concrete,
            zonos_v1_engine,
            zonos_engine,
            irodori_engine,
            custom_default_engine,
            enrolled_voices: Mutex::new(HashMap::new()),
        }
    }

    pub fn current_settings(&self) -> Arc<CompiledSettings> {
        self.settings
            .read()
            .expect("settings RwLock poisoned")
            .clone()
    }

    pub fn default_engine(&self, cfg: &CompiledSettings) -> Arc<dyn TtsEngine> {
        if let Some(ref custom) = self.custom_default_engine {
            return custom.clone();
        }
        match cfg.raw.general.default_engine.as_str() {
            "zonos" => self.zonos_v1_engine.clone(),
            "irodori" => self.irodori_engine.clone(),
            _ => self.zonos_engine.clone(),
        }
    }

    /// Dynamically resolves the target TTS engine based on the OpenAI `model` parameter.
    ///
    /// - `None`, `""`, `"default"`, `"tts-1"`, `"tts-1-hd"`, `"gpt-4o-mini-tts"` -> Default engine (`general.default_engine`)
    /// - `"zonos"`, `"zonos-v0.1"`, `"zonos1"`, `"zonos-1"` -> `self.zonos_v1_engine`
    /// - `"zonos2"`, `"zonos-2"` -> `self.zonos_engine`
    /// - `"irodori"`, `"irodori-tts"`, `"irodori*"` -> `self.irodori_engine`
    pub fn resolve_engine(&self, model: Option<&str>) -> Result<Arc<dyn TtsEngine>, AppError> {
        let cfg = self.current_settings();
        self.resolve_engine_with_settings(model, &cfg)
    }

    pub fn resolve_engine_with_settings(
        &self,
        model: Option<&str>,
        cfg: &CompiledSettings,
    ) -> Result<Arc<dyn TtsEngine>, AppError> {
        let Some(raw) = model else {
            return Ok(self.default_engine(cfg));
        };
        let m = raw.trim().to_lowercase();
        if m.is_empty()
            || m == "default"
            || m == "tts-1"
            || m == "tts-1-hd"
            || m == "gpt-4o-mini-tts"
        {
            return Ok(self.default_engine(cfg));
        }
        if m == "irodori"
            || m == "irodori-tts"
            || m.starts_with("irodori")
            || m == cfg.raw.engines.irodori.model.trim().to_lowercase()
        {
            return Ok(self.irodori_engine.clone());
        }
        if m == "zonos2" || m == "zonos-2" || m == "zonos_2" {
            return Ok(self.zonos_engine.clone());
        }
        if m == "zonos"
            || m == "zonos-v0.1"
            || m == "zonos_v1"
            || m == "zonos1"
            || m == "zonos-1"
        {
            return Ok(self.zonos_v1_engine.clone());
        }
        if m == self.config.default_model.trim().to_lowercase() {
            return Ok(self.default_engine(cfg));
        }

        Err(AppError::BadRequest(format!(
            "Unsupported TTS model: '{raw}'. Supported models: 'zonos' (Zonos v0.1), 'zonos2' (Zonos 2), 'irodori-tts' ('irodori'), 'tts-1', 'tts-1-hd'"
        )))
    }
}
