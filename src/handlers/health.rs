use crate::handlers::AppState;
use axum::{extract::State, Json};
use serde_json::{json, Value};
use std::sync::Arc;

pub async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let cfg = state.current_settings();
    let default_eng = state.default_engine(&cfg);

    let (zonos_v1_res, zonos2_res, irodori_res) = tokio::join!(
        state.zonos_v1_engine.health_check(),
        state.zonos_engine.health_check(),
        state.irodori_engine.health_check()
    );

    let zonos_v1_connected = zonos_v1_res.is_ok();
    let zonos2_connected = zonos2_res.is_ok();
    let irodori_connected = irodori_res.is_ok();

    let engine_connected = match default_eng.name() {
        "zonos" => zonos_v1_connected,
        "zonos2" => zonos2_connected,
        "irodori" => irodori_connected,
        _ => default_eng.health_check().await.is_ok(),
    };

    Json(json!({
        "status": "ok",
        "engine": default_eng.name(),
        "engine_connected": engine_connected,
        "zonos_v1_connected": zonos_v1_connected,
        "zonos_connected": zonos2_connected,
        "irodori_connected": irodori_connected,
        "engines": {
            "zonos": {
                "connected": zonos_v1_connected,
                "url": &cfg.raw.engines.zonos.url,
                "model": state.zonos_v1_engine.default_model(),
            },
            "zonos2": {
                "connected": zonos2_connected,
                "url": &cfg.raw.engines.zonos2.url,
                "model": state.zonos_engine.default_model(),
            },
            "irodori": {
                "connected": irodori_connected,
                "url": &cfg.raw.engines.irodori.url,
                "model": &cfg.raw.engines.irodori.model,
            }
        },
        "default_voice": &cfg.raw.general.default_voice,
        "default_language": &cfg.raw.general.default_language,
        "default_model": default_eng.default_model(),
    }))
}
