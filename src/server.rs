use crate::config::Config;
use crate::engine::create_engine;
use crate::handlers::{
    handle_fallback, handle_health, handle_models, handle_speech, handle_upload_voice, AppState,
};
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

pub async fn run_server(config: Config) -> anyhow::Result<()> {
    // Validate default engine configuration at startup
    let default_engine = create_engine(&config)?;
    let bind_addr = format!("{}:{}", config.host, config.port);

    info!(
        addr = %bind_addr,
        default_engine = %default_engine.name(),
        irodori_url = %config.irodori_url,
        zonos_url = %config.zonos_url,
        default_voice = %config.default_voice,
        default_model = %default_engine.default_model(),
        "Starting openai-tts-bridge server (multi-engine dynamic routing enabled)"
    );

    let state = Arc::new(AppState::with_engine(config.clone(), default_engine));
    let app = create_router(state);

    let listener = TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
