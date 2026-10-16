use axum::{
    http::{Method, StatusCode, Uri},
    response::IntoResponse,
};
use tracing::warn;

pub mod health;
pub mod models;
pub mod speech;
pub mod state;

pub use health::handle_health;
pub use models::handle_models;
pub use speech::{handle_speech, handle_upload_voice};
pub use state::AppState;

/// Fallback handler for unmapped routes, printing a visible warning with method and URI
pub async fn handle_fallback(method: Method, uri: Uri) -> impl IntoResponse {
    warn!(%method, %uri, "Received request to unmapped route (404)");
    (StatusCode::NOT_FOUND, format!("Not Found: {method} {uri}"))
}
