pub mod gradio;
pub mod health;
pub mod models;
pub mod speech;
pub mod state;

pub use gradio::{
    handle_fallback, handle_gradio_call_generate_audio, handle_gradio_file, handle_gradio_sse,
    handle_gradio_upload, handle_languages, handle_speakers,
};
pub use health::handle_health;
pub use models::handle_models;
pub use speech::{handle_speech, handle_upload_voice};
pub use state::{AppState, GradioEventStatus};
