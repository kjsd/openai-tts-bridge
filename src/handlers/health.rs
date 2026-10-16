use crate::handlers::AppState;
use axum::{extract::State, Json};
use serde_json::{json, Value};
use std::sync::Arc;

pub async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let (default_res, irodori_res, zonos_res) = tokio::join!(
        state.engine.health_check(),
        state.irodori_engine.health_check(),
        state.zonos_engine.health_check()
    );

    let engine_connected = default_res.is_ok();
    let irodori_connected = irodori_res.is_ok();
    let zonos_connected = zonos_res.is_ok();

    Json(json!({
        "status": "ok",
        "engine": state.engine.name(),
        "engine_connected": engine_connected,
        "zonos_connected": zonos_connected,
        "irodori_connected": irodori_connected,
        "engines": {
            "irodori": {
                "connected": irodori_connected,
                "url": &state.config.irodori_url,
                "model": state.irodori_engine.default_model(),
            },
            "zonos2": {
                "connected": zonos_connected,
                "url": &state.config.zonos_url,
                "model": state.zonos_engine.default_model(),
            }
        },
        "default_voice": &state.config.default_voice,
        "default_model": state.engine.default_model(),
    }))
}
