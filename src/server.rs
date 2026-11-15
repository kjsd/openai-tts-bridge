use crate::config::Config;
use crate::handlers::{
    handle_admin_ui, handle_fallback, handle_get_settings, handle_health, handle_models,
    handle_preview, handle_put_settings, handle_speech, handle_upload_voice, AppState,
};
use crate::settings::normalize_engine_key;
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;

pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let body_limit_bytes = state.config.max_body_size_mb * 1024 * 1024;

    Router::new()
        // Embedded Management Web UI & Admin API
        .route("/", get(handle_admin_ui))
        .route("/admin", get(handle_admin_ui))
        .route(
            "/api/settings",
            get(handle_get_settings).put(handle_put_settings),
        )
        .route("/api/preview", post(handle_preview))
        // OpenAI-compatible TTS endpoints
        .route("/v1/audio/speech", post(handle_speech))
        .route("/v1/audio/voices", post(handle_upload_voice))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        // Fallback for unmapped routes
        .fallback(handle_fallback)
        .layer(axum::extract::DefaultBodyLimit::max(body_limit_bytes))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(mut config: Config) -> anyhow::Result<()> {
    if normalize_engine_key(&config.tts_engine).is_none() {
        anyhow::bail!(
            "Unsupported TTS_ENGINE '{}'. Must be 'zonos', 'zonos2', or 'irodori'.",
            config.tts_engine
        );
    }

    // Default persistent TOML path when running as a standalone server
    if config.config_file.is_none() {
        config.config_file = Some("config.toml".to_string());
    }

    let bind_addr = format!("{}:{}", config.host, config.port);
    let state = Arc::new(AppState::new(config.clone()));
    let cfg = state.current_settings();

    info!(
        addr = %bind_addr,
        config_file = ?state.config_file,
        default_engine = %cfg.raw.general.default_engine,
        zonos_v1_url = %cfg.raw.engines.zonos.url,
        zonos2_url = %cfg.raw.engines.zonos2.url,
        irodori_url = %cfg.raw.engines.irodori.url,
        default_voice = %cfg.raw.general.default_voice,
        dict_entries = cfg.raw.dictionary.len(),
        "Starting openai-tts-bridge v0.3 unified server (Web UI available at http://{}/)",
        bind_addr
    );

    let app = create_router(state);

    let listener = TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
