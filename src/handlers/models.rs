use crate::handlers::AppState;
use axum::{extract::State, Json};
use serde_json::{json, Value};
use std::sync::Arc;

pub async fn handle_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let irodori_model = state.irodori_engine.default_model();
    let zonos_model = state.zonos_engine.default_model();

    let mut models = vec![
        json!({
            "id": irodori_model,
            "object": "model",
            "created": 1725400000,
            "owned_by": "irodori"
        }),
        json!({
            "id": zonos_model,
            "object": "model",
            "created": 1725400000,
            "owned_by": "zonos2"
        }),
    ];

    if irodori_model != "irodori" {
        models.push(json!({
            "id": "irodori",
            "object": "model",
            "created": 1725400000,
            "owned_by": "irodori"
        }));
    }

    if zonos_model != "zonos2" {
        models.push(json!({
            "id": "zonos2",
            "object": "model",
            "created": 1725400000,
            "owned_by": "zonos2"
        }));
    }

    models.push(json!({
        "id": "tts-1",
        "object": "model",
        "created": 1725400000,
        "owned_by": "openai"
    }));
    models.push(json!({
        "id": "tts-1-hd",
        "object": "model",
        "created": 1725400000,
        "owned_by": "openai"
    }));

    Json(json!({
        "object": "list",
        "data": models
    }))
}
