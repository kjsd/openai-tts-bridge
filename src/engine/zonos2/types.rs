use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct ZonosGenerateRequest {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_embedding_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_audio_base64: Option<String>,
    pub language: String,
    pub emotion_enabled: bool,
    pub emotion_sliders: HashMap<String, f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion_valence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion_arousal: Option<f32>,
    pub emotion_cfg_scale: f32,
    pub speed: f32,
    pub speaking_rate_enabled: bool,
    pub accurate_mode: bool,
    pub stream: bool,
}
