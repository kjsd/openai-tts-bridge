use axum::{
    extract::State,
    response::Html,
    Json,
};
use serde::Deserialize;
use std::sync::Arc;
use tracing::info;

use crate::engine::{PreviewResult, SynthesisRequest};
use crate::error::AppError;
use crate::handlers::AppState;
use crate::settings::{CompiledSettings, RuntimeSettings};

const ADMIN_HTML: &str = include_str!("../webui/index.html");

#[derive(Debug, Deserialize)]
pub struct PreviewRequest {
    pub model: Option<String>,
    pub input: String,
    pub voice: Option<String>,
    pub speed: Option<f32>,
    pub language: Option<String>,
}

/// GET / and GET /admin
/// Serves the embedded single-page management UI compiled directly into the Rust binary.
pub async fn handle_admin_ui() -> Html<&'static str> {
    Html(ADMIN_HTML)
}

/// GET /api/settings
/// Returns the active runtime configuration and pronunciation dictionary as JSON.
pub async fn handle_get_settings(
    State(state): State<Arc<AppState>>,
) -> Json<RuntimeSettings> {
    let cfg = state.current_settings();
    Json(cfg.raw.clone())
}

/// PUT /api/settings
/// Validates, compiles, atomically writes to `config.toml` (when configured),
/// and hot-swaps the in-memory `CompiledSettings` snapshot without restarting.
pub async fn handle_put_settings(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RuntimeSettings>,
) -> Result<Json<RuntimeSettings>, AppError> {
    let compiled = CompiledSettings::new(payload)?;

    if let Some(ref path) = state.config_file {
        compiled.raw.save_atomic(path)?;
        info!(path = %path.display(), "Persisted updated settings atomically to TOML");
    }

    let prev = state.current_settings();
    let zonos_v1_url_changed = prev.raw.engines.zonos.url != compiled.raw.engines.zonos.url;
    let irodori_url_changed = prev.raw.engines.irodori.url != compiled.raw.engines.irodori.url;

    let response_raw = compiled.raw.clone();
    {
        let mut guard = state.settings.write().expect("settings RwLock poisoned");
        *guard = Arc::new(compiled);
    }

    if zonos_v1_url_changed {
        state.zonos_v1_concrete.clear_voice_cache().await;
    }
    if irodori_url_changed {
        state.irodori_concrete.clear_voice_cache().await;
    }

    info!(
        default_engine = %response_raw.general.default_engine,
        dict_entries = response_raw.dictionary.len(),
        "Hot-reloaded runtime settings in memory"
    );

    Ok(Json(response_raw))
}

/// POST /api/preview
/// Dry-runs tag parsing, dictionary substitution, and engine-specific text normalization
/// without invoking any GPU TTS backend.
pub async fn handle_preview(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PreviewRequest>,
) -> Result<Json<PreviewResult>, AppError> {
    let cfg = state.current_settings();
    let engine = state.resolve_engine_with_settings(payload.model.as_deref(), &cfg)?;

    let req = SynthesisRequest {
        text: payload.input,
        voice: payload.voice,
        speed: payload.speed,
        language: payload.language,
        ..SynthesisRequest::default()
    };

    let preview = engine.preview(&req, &cfg)?;
    Ok(Json(preview))
}
