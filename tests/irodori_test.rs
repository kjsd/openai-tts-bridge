use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use base64::Engine;
use http_body_util::BodyExt;
use openai_tts_bridge::{create_engine, create_router, AppState, Config};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn create_irodori_test_config(irodori_url: String) -> Config {
    Config {
        tts_engine: "irodori".to_string(),
        host: "127.0.0.1".to_string(),
        port: 8000,
        zonos_url: "http://127.0.0.1:1919".to_string(),
        irodori_url,
        irodori_model: "irodori-tts".to_string(),
        default_voice: "default".to_string(),
        default_model: "irodori-tts".to_string(),
        log_level: "error".to_string(),
        max_body_size_mb: 100,
        default_speed: 1.0,
    }
}

// Generate dummy 16-bit PCM WAV bytes for mock responses
fn make_dummy_wav() -> Vec<u8> {
    let mut header = Vec::new();
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&36u32.to_le_bytes());
    header.extend_from_slice(b"WAVE");
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes()); // PCM
    header.extend_from_slice(&1u16.to_le_bytes()); // Mono
    header.extend_from_slice(&44100u32.to_le_bytes()); // Sample rate
    header.extend_from_slice(&88200u32.to_le_bytes()); // Byte rate
    header.extend_from_slice(&2u16.to_le_bytes()); // Block align
    header.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample
    header.extend_from_slice(b"data");
    header.extend_from_slice(&0u32.to_le_bytes());
    header
}

#[tokio::test]
async fn test_irodori_models_endpoint() {
    let config = create_irodori_test_config("http://127.0.0.1:9999".to_string());
    let engine = create_engine(&config).unwrap();
    let state = Arc::new(AppState::with_engine(config, engine));
    let app = create_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let models = json["data"].as_array().unwrap();
    assert!(models.iter().any(|m| m["id"] == "irodori-tts"));
}

#[tokio::test]
async fn test_irodori_speech_tag_to_emoji_conversion() {
    let mock_server = MockServer::start().await;
    let dummy_wav = make_dummy_wav();

    // Verify that the mock server receives text with [whisper] converted to 👂 emoji
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(move |req: &wiremock::Request| {
            let body_json: Value = serde_json::from_slice(&req.body).unwrap();
            let input_text = body_json["input"].as_str().unwrap();
            assert!(input_text.contains('👂'), "Text must contain whisper emoji");
            assert!(input_text.contains("パパ、寒くない？"));
            assert!(!input_text.contains("[whisper]"));

            ResponseTemplate::new(200)
                .set_body_bytes(dummy_wav.clone())
                .insert_header("content-type", "audio/wav")
        })
        .mount(&mock_server)
        .await;

    let config = create_irodori_test_config(mock_server.uri());
    let engine = create_engine(&config).unwrap();
    let state = Arc::new(AppState::with_engine(config, engine));
    let app = create_router(state);

    let speech_req = json!({
        "model": "irodori-tts",
        "input": "[whisper] パパ、寒くない？",
        "voice": "default"
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&speech_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "audio/wav"
    );
    assert_eq!(
        response.headers().get("x-emotion-tags").unwrap(),
        "whisper"
    );
}

#[tokio::test]
async fn test_irodori_speech_missing_bracket_recovery() {
    let mock_server = MockServer::start().await;
    let dummy_wav = make_dummy_wav();

    // Verify that "chuckle] さあ..." is recovered to 🤭 emoji
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(move |req: &wiremock::Request| {
            let body_json: Value = serde_json::from_slice(&req.body).unwrap();
            let input_text = body_json["input"].as_str().unwrap();
            assert!(input_text.contains('🤭'), "Text must contain chuckle emoji");
            assert!(input_text.contains("さあケンジ君"));
            assert!(!input_text.contains("chuckle]"));

            ResponseTemplate::new(200)
                .set_body_bytes(dummy_wav.clone())
                .insert_header("content-type", "audio/wav")
        })
        .mount(&mock_server)
        .await;

    let config = create_irodori_test_config(mock_server.uri());
    let engine = create_engine(&config).unwrap();
    let state = Arc::new(AppState::with_engine(config, engine));
    let app = create_router(state);

    let speech_req = json!({
        "model": "irodori-tts",
        "input": "chuckle] さあケンジ君、やっとウィンターホールドに着いたわ。",
        "voice": "default"
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&speech_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-emotion-tags").unwrap(),
        "chuckle"
    );
}

#[tokio::test]
async fn test_irodori_speech_official_sound_and_expression_tags() {
    let mock_server = MockServer::start().await;
    let dummy_wav = make_dummy_wav();

    // Verify clear throat -> 🤧 and pant -> 🌬️
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(move |req: &wiremock::Request| {
            let body_json: Value = serde_json::from_slice(&req.body).unwrap();
            let input_text = body_json["input"].as_str().unwrap();
            assert!(input_text.contains('🤧'), "Text must contain sneeze/throat-clear emoji");
            assert!(input_text.contains('🌬'), "Text must contain pant/breath emoji");
            assert!(input_text.contains("走って逃げてきたの"));

            ResponseTemplate::new(200)
                .set_body_bytes(dummy_wav.clone())
                .insert_header("content-type", "audio/wav")
        })
        .mount(&mock_server)
        .await;

    let config = create_irodori_test_config(mock_server.uri());
    let engine = create_engine(&config).unwrap();
    let state = Arc::new(AppState::with_engine(config, engine));
    let app = create_router(state);

    let speech_req = json!({
        "model": "irodori-tts",
        "input": "[clear throat] [pant] 走って逃げてきたの…！",
        "voice": "default"
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&speech_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-emotion-tags").unwrap(),
        "clear throat, pant"
    );
}

#[tokio::test]
async fn test_irodori_speech_with_custom_speaker_audio_upload() {
    let mock_server = MockServer::start().await;
    let dummy_wav = make_dummy_wav();
    let dummy_ref_audio = b"dummy_reference_voice_data_for_irodori";
    let dummy_b64 = base64::prelude::BASE64_STANDARD.encode(dummy_ref_audio);
    let expected_hash = format!("{:x}", md5::compute(dummy_ref_audio));
    let expected_voice_id = format!("custom_ref_{expected_hash}");

    // 1. Mock voice upload endpoint: POST /v1/audio/voices -> 201 Created
    Mock::given(method("POST"))
        .and(path("/v1/audio/voices"))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "voice_id": expected_voice_id,
            "path": format!("/voices/{expected_voice_id}.wav")
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 2. Mock speech synthesis endpoint: POST /v1/audio/speech with resolved dynamic voice
    let expected_voice_clone = expected_voice_id.clone();
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(move |req: &wiremock::Request| {
            let body_json: Value = serde_json::from_slice(&req.body).unwrap();
            assert_eq!(body_json["voice"].as_str().unwrap(), expected_voice_clone);

            ResponseTemplate::new(200)
                .set_body_bytes(dummy_wav.clone())
                .insert_header("content-type", "audio/wav")
        })
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = create_irodori_test_config(mock_server.uri());
    let engine = create_engine(&config).unwrap();
    let state = Arc::new(AppState::with_engine(config, engine));
    let app = create_router(state);

    let speech_req = json!({
        "model": "irodori-tts",
        "input": "こんにちは、テストです。",
        "speaker_audio_base64": dummy_b64
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&speech_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_irodori_speech_with_custom_speaker_audio_cache_hit() {
    let mock_server = MockServer::start().await;
    let dummy_wav = make_dummy_wav();
    let dummy_ref_audio = b"cached_reference_voice_data";
    let dummy_b64 = base64::prelude::BASE64_STANDARD.encode(dummy_ref_audio);
    let expected_hash = format!("{:x}", md5::compute(dummy_ref_audio));
    let expected_voice_id = format!("custom_ref_{expected_hash}");

    // 1. Mock voice upload endpoint returning 409 Conflict (already exists)
    Mock::given(method("POST"))
        .and(path("/v1/audio/voices"))
        .respond_with(ResponseTemplate::new(409).set_body_json(json!({
            "detail": format!("Voice '{expected_voice_id}' already exists.")
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 2. Mock speech synthesis endpoint
    let expected_voice_clone = expected_voice_id.clone();
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(move |req: &wiremock::Request| {
            let body_json: Value = serde_json::from_slice(&req.body).unwrap();
            assert_eq!(body_json["voice"].as_str().unwrap(), expected_voice_clone);

            ResponseTemplate::new(200)
                .set_body_bytes(dummy_wav.clone())
                .insert_header("content-type", "audio/wav")
        })
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = create_irodori_test_config(mock_server.uri());
    let engine = create_engine(&config).unwrap();
    let state = Arc::new(AppState::with_engine(config, engine));
    let app = create_router(state);

    let speech_req = json!({
        "model": "irodori-tts",
        "input": "こんにちは、キャッシュテストです。",
        "speaker_audio_base64": dummy_b64
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&speech_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_irodori_skyrimnet_openai_voice_enrollment_and_synthesis() {
    let mock_server = MockServer::start().await;
    let dummy_wav = make_dummy_wav();
    let skyrim_voice_id = "snv1-a1b2c3d4e5f6";
    let skyrim_local_voice = "voice:malebrute";
    let ref_wav_bytes = b"RIFF_skyrim_malebrute_wav_bytes";
    let expected_hash = format!("{:x}", md5::compute(ref_wav_bytes));
    let expected_irodori_voice_id = format!("custom_ref_{expected_hash}");

    // 1. Mock Irodori backend voice upload
    Mock::given(method("POST"))
        .and(path("/v1/audio/voices"))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "voice_id": expected_irodori_voice_id,
            "path": format!("/voices/{expected_irodori_voice_id}.wav")
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 2. Mock Irodori backend speech synthesis
    let expected_irodori_voice_clone = expected_irodori_voice_id.clone();
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(move |req: &wiremock::Request| {
            let body_json: Value = serde_json::from_slice(&req.body).unwrap();
            assert_eq!(
                body_json["voice"].as_str().unwrap(),
                expected_irodori_voice_clone
            );
            let input_text = body_json["input"].as_str().unwrap();
            assert!(input_text.contains('👂')); // [conspiratorial] -> 👂
            assert!(input_text.contains("ドラゴンボーン、聞いて"));

            ResponseTemplate::new(200)
                .set_body_bytes(dummy_wav.clone())
                .insert_header("content-type", "audio/wav")
        })
        .expect(1)
        .mount(&mock_server)
        .await;

    let config = create_irodori_test_config(mock_server.uri());
    let engine = create_engine(&config).unwrap();
    let state = Arc::new(AppState::with_engine(config, engine));
    let app = create_router(state);

    // Step A: SkyrimNet calls POST /v1/audio/speech with `voice:malebrute` (or `snv1-...`)
    // after a bridge restart before re-enrolling -> Expect HTTP 404 with error.code == "voice_not_found"
    let uncached_local_req = json!({
        "model": "irodori-tts",
        "input": "[conspiratorial] ドラゴンボーン、聞いて",
        "voice": skyrim_local_voice,
        "response_format": "wav",
        "speed": 1.0
    });

    let uncached_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&uncached_local_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(uncached_resp.status(), StatusCode::NOT_FOUND);
    let err_body = uncached_resp.into_body().collect().await.unwrap().to_bytes();
    let err_json: Value = serde_json::from_slice(&err_body).unwrap();
    assert_eq!(err_json["error"]["code"], "voice_not_found");

    // Step B: SkyrimNet uploads reference voice via POST /v1/audio/voices
    // Exact format sent by SkyrimNet vbeta27: title = "voice:malebrute", ref_audio filename = "snv1-<sha256>.wav"
    let boundary = "------------------------skyrimnetboundary";
    let multipart_body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"ref_text\"\r\n\r\n\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\n{skyrim_local_voice}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"ref_audio\"; filename=\"{skyrim_voice_id}.wav\"\r\nContent-Type: audio/wav\r\n\r\n{}\r\n--{boundary}--\r\n",
        std::str::from_utf8(ref_wav_bytes).unwrap()
    );

    let enroll_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/audio/voices")
                .method("POST")
                .header(
                    "Content-Type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(multipart_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(enroll_resp.status(), StatusCode::OK);
    let enroll_body = enroll_resp.into_body().collect().await.unwrap().to_bytes();
    let enroll_json: Value = serde_json::from_slice(&enroll_body).unwrap();
    assert_eq!(enroll_json["voice_id"], skyrim_voice_id);

    // Step C: SkyrimNet retries POST /v1/audio/speech with the enrolled voice_id
    let retry_req = json!({
        "model": "irodori-tts",
        "input": "[conspiratorial] ドラゴンボーン、聞いて",
        "voice": skyrim_voice_id,
        "response_format": "wav",
        "speed": 1.0
    });

    let retry_resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&retry_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(retry_resp.status(), StatusCode::OK);
    assert_eq!(
        retry_resp.headers().get("x-emotion-tags").unwrap(),
        "conspiratorial"
    );
}
