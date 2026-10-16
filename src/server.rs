use crate::config::Config;
use crate::engine::create_engine;
use crate::handlers::{
    handle_fallback, handle_gradio_call_generate_audio, handle_gradio_file, handle_gradio_sse,
    handle_gradio_upload, handle_health, handle_languages, handle_models, handle_speakers,
    handle_speech, handle_upload_voice, AppState,
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
        // Standard OpenAI endpoints
        .route("/v1/audio/speech", post(handle_speech))
        .route("/v1/audio/voices", post(handle_upload_voice))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        // Gradio API endpoints (SkyrimNet / Chatterbox compatible)
        .route("/gradio_api/upload", post(handle_gradio_upload))
        .route("/gradio_api/call/generate_audio", post(handle_gradio_call_generate_audio))
        .route("/gradio_api/call/generate_audio/{event_id}", get(handle_gradio_sse))
        .route("/gradio_api/{*path}", get(handle_gradio_file).head(handle_gradio_file))
        // SkyrimNet / XTTS compatible speaker & language endpoints
        .route("/speakers", get(handle_speakers))
        .route("/speakers_list", get(handle_speakers))
        .route("/get_speakers", get(handle_speakers))
        .route("/languages", get(handle_languages))
        // Fallback for unmapped routes
        .fallback(handle_fallback)
        .layer(axum::extract::DefaultBodyLimit::max(body_limit_bytes))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(config: Config) -> anyhow::Result<()> {
    let engine = create_engine(&config)?;
    let bind_addr = format!("{}:{}", config.host, config.port);

    info!(
        addr = %bind_addr,
        engine = %engine.name(),
        default_voice = %config.default_voice,
        default_model = %engine.default_model(),
        "Starting openai-tts-bridge server"
    );

    let state = Arc::new(AppState::with_engine(config.clone(), engine));
    let app = create_router(state);

    let listener = TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
