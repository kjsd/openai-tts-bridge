use crate::config::Config;
use crate::engine::{create_engine, TtsEngine, ZonosEngine};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AppState {
    pub config: Config,
    pub engine: Arc<dyn TtsEngine>,
    pub enrolled_voices: Mutex<HashMap<String, Vec<u8>>>,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let engine = create_engine(&config).unwrap_or_else(|_| {
            Arc::new(ZonosEngine::new(&config).expect("Failed to initialize ZonosEngine"))
        });
        Self {
            config,
            engine,
            enrolled_voices: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_engine(config: Config, engine: Arc<dyn TtsEngine>) -> Self {
        Self {
            config,
            engine,
            enrolled_voices: Mutex::new(HashMap::new()),
        }
    }
}
