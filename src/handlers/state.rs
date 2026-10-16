use crate::config::Config;
use crate::engine::{create_engine, TtsEngine, ZonosEngine};
use crate::zonos::ZonosClient;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub enum GradioEventStatus {
    Generating,
    Done(Value),
    Error(String),
}

pub struct AppState {
    pub config: Config,
    pub engine: Arc<dyn TtsEngine>,
    pub zonos: ZonosClient,
    pub gradio_events: Mutex<HashMap<String, GradioEventStatus>>,
    pub enrolled_voices: Mutex<HashMap<String, Vec<u8>>>,
}

impl AppState {
    pub fn new(config: Config, zonos: ZonosClient) -> Self {
        let engine = create_engine(&config).unwrap_or_else(|_| {
            Arc::new(ZonosEngine::new(&config).expect("Failed to initialize ZonosEngine"))
        });
        Self {
            config,
            engine,
            zonos,
            gradio_events: Mutex::new(HashMap::new()),
            enrolled_voices: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_engine(config: Config, engine: Arc<dyn TtsEngine>) -> Self {
        let zonos = ZonosClient::new(config.zonos_url.clone());
        Self {
            config,
            engine,
            zonos,
            gradio_events: Mutex::new(HashMap::new()),
            enrolled_voices: Mutex::new(HashMap::new()),
        }
    }
}
