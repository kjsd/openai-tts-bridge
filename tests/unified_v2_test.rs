use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use base64::Engine as _;
use byteorder::{LittleEndian, WriteBytesExt};
use http_body_util::BodyExt;
use openai_tts_bridge::{create_router, AppState, Config, RuntimeSettings};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_pcm16_wav(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let riff_len = 36 + data_len;
    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_len.to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM16
    buf.extend_from_slice(&1u16.to_le_bytes()); // Mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}

#[tokio::test]
async fn test_zonos_v1_gradio_e2e_chunking_upload_cache_and_wav_concat() {
    let mock_gradio = MockServer::start().await;

    // 1. Mock POST /gradio_api/upload -> must only be called ONCE for 2 sentence chunks!
    Mock::given(method("POST"))
        .and(path("/gradio_api/upload"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            "/tmp/gradio/abc123/speaker_ref.wav"
        ])))
        .expect(1)
        .mount(&mock_gradio)
        .await;

    // 2. Mock POST /gradio_api/call/generate_audio -> called TWICE (once per sentence chunk)
    let call_counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = call_counter.clone();
    Mock::given(method("POST"))
        .and(path("/gradio_api/call/generate_audio"))
        .respond_with(move |req: &wiremock::Request| {
            let idx = counter_clone.fetch_add(1, Ordering::SeqCst);
            let payload: Value = serde_json::from_slice(&req.body).unwrap();
            let data = payload["data"].as_array().expect("29-arg Gradio data array");
            assert_eq!(data.len(), 29);
            assert_eq!(data[0], "Zyphra/Zonos-v0.1-hybrid");
            assert_eq!(data[2], "ja");
            assert_eq!(data[3]["path"], "/tmp/gradio/abc123/speaker_ref.wav");

            // Check chunk text normalization & terminal punctuation
            if idx == 0 {
                assert_eq!(data[1], "ひがししらかわむらへ行こう！");
            } else {
                assert_eq!(data[1], "エムボットツーが10センチ進んだよ。");
            }

            // [excited] boosts happiness (0.05 + 0.5 = 0.55)
            let happiness = data[5].as_f64().unwrap();
            assert!(happiness > 0.5, "Expected boosted happiness, got {happiness}");

            ResponseTemplate::new(200).set_body_json(json!({
                "event_id": format!("evt_{idx}")
            }))
        })
        .expect(2)
        .mount(&mock_gradio)
        .await;

    // 3. Mock SSE endpoints GET /gradio_api/call/generate_audio/evt_0 & evt_1
    let base_uri = mock_gradio.uri();
    for idx in 0..2 {
        let sse_body = format!(
            "event: generating\ndata: null\n\nevent: complete\ndata: [{{\"url\": \"{base_uri}/gradio_api/file=/tmp/out_{idx}.wav\"}}, 42]\n\n"
        );
        Mock::given(method("GET"))
            .and(path(format!("/gradio_api/call/generate_audio/evt_{idx}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .expect(1)
            .mount(&mock_gradio)
            .await;

        // Each chunk is 2 samples at 1000 Hz so 100ms silence = 100 samples
        let wav_bytes = make_pcm16_wav(&[1000 + idx as i16, -1000], 1000);
        Mock::given(method("GET"))
            .and(path(format!("/gradio_api/file=/tmp/out_{idx}.wav")))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/wav")
                    .set_body_bytes(wav_bytes),
            )
            .expect(1)
            .mount(&mock_gradio)
            .await;
    }

    let config = Config {
        tts_engine: "zonos".to_string(),
        zonos_v1_url: mock_gradio.uri(),
        ..Config::default()
    };
    let state = Arc::new(AppState::new(config));
    let app = create_router(state);

    let ref_b64 = base64::engine::general_purpose::STANDARD.encode(b"RIFF_fake_ref_voice");
    let req_body = json!({
        "model": "zonos",
        "input": "[excited] 東白川村へ行こう…！ mBot2が10cm進んだよ",
        "speaker_audio_base64": ref_b64,
        "language": "ja"
    });

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get("x-emotion-tags").unwrap(), "excited");

    let wav_out = resp.into_body().collect().await.unwrap().to_bytes();
    // 2 samples (chunk 0) + 100 silence samples (100ms @ 1000Hz) + 2 samples (chunk 1) = 104 samples
    // 44 header bytes + 104 * 2 bytes = 252 bytes
    assert_eq!(wav_out.len(), 44 + 104 * 2);
}

#[tokio::test]
async fn test_per_engine_normalization_contrast_zonos2_vs_irodori() {
    let mock_zonos2 = MockServer::start().await;
    let mock_irodori = MockServer::start().await;

    // Zonos 2 MUST convert dictionary + units + wave dash (`〜` -> `ー`) + strip trailing `…`
    let mut fake_pcm = Vec::new();
    fake_pcm.write_f32::<LittleEndian>(0.25).unwrap();
    Mock::given(method("POST"))
        .and(path("/tts/generate"))
        .respond_with(move |req: &wiremock::Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap();
            assert_eq!(
                body["text"].as_str().unwrap(),
                "ひがししらかわむらでのーんびり10センチ進もう"
            );
            ResponseTemplate::new(200).set_body_bytes(fake_pcm.clone())
        })
        .expect(1)
        .mount(&mock_zonos2)
        .await;

    // Irodori MUST apply dictionary (`東白川村`), convert `[whisper]` -> `👂`,
    // and PRESERVE `〜`, `10cm`, and trailing `…` untouched!
    let dummy_wav = make_pcm16_wav(&[0, 100], 24000);
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(move |req: &wiremock::Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap();
            assert_eq!(
                body["input"].as_str().unwrap(),
                "👂 ひがししらかわむらでの〜んびり10cm進もう…"
            );
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/wav")
                .set_body_bytes(dummy_wav.clone())
        })
        .expect(1)
        .mount(&mock_irodori)
        .await;

    let config = Config {
        zonos_url: mock_zonos2.uri(),
        irodori_url: mock_irodori.uri(),
        ..Config::default()
    };
    let app = create_router(Arc::new(AppState::new(config)));

    let input_text = "[whisper] 東白川村での〜んびり10cm進もう…";

    // 1. Send to zonos2
    let resp_z2 = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({ "model": "zonos2", "input": input_text })).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_z2.status(), StatusCode::OK);

    // 2. Send identical input to irodori-tts
    let resp_iro = app
        .oneshot(
            Request::builder()
                .uri("/v1/audio/speech")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({ "model": "irodori-tts", "input": input_text }))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_iro.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_webui_settings_hot_reload_toml_persistence_and_preview() {
    let tmp_toml = std::env::temp_dir().join(format!(
        "openai_tts_bridge_v2_e2e_{}.toml",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp_toml);

    let config = Config {
        tts_engine: "zonos2".to_string(),
        config_file: Some(tmp_toml.to_string_lossy().to_string()),
        ..Config::default()
    };
    let app = create_router(Arc::new(AppState::new(config.clone())));

    // 1. Verify embedded Web UI HTML is served at GET /
    let ui_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ui_resp.status(), StatusCode::OK);
    let html_bytes = ui_resp.into_body().collect().await.unwrap().to_bytes();
    let html_str = std::str::from_utf8(&html_bytes).unwrap();
    assert!(html_str.contains("openai-tts-bridge"));

    // 2. GET /api/settings -> modify default_engine to "irodori" & add custom dictionary term -> PUT /api/settings
    let get_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/settings")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(get_resp.status(), StatusCode::OK);
    let mut settings: RuntimeSettings = serde_json::from_slice(
        &get_resp.into_body().collect().await.unwrap().to_bytes(),
    )
    .unwrap();
    assert_eq!(settings.general.default_engine, "zonos2");

    settings.general.default_engine = "irodori".to_string();
    settings.dictionary.push(openai_tts_bridge::normalizer::DictionaryEntry {
        surface: "賢嗣".to_string(),
        reading: "けんじ".to_string(),
        case_sensitive: true,
        enabled: true,
    });

    let put_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/settings")
                .method("PUT")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&settings).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(put_resp.status(), StatusCode::OK);
    assert!(tmp_toml.exists(), "config.toml must be written atomically");

    // 3. POST /api/preview with model="default" MUST immediately use hot-reloaded "irodori" + new dictionary!
    let preview_resp = app
        .oneshot(
            Request::builder()
                .uri("/api/preview")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "default",
                        "input": "[chuckle] 賢嗣、大好き〜！"
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(preview_resp.status(), StatusCode::OK);
    let preview_json: Value = serde_json::from_slice(
        &preview_resp.into_body().collect().await.unwrap().to_bytes(),
    )
    .unwrap();
    assert_eq!(preview_json["engine"], "irodori");
    assert_eq!(preview_json["processed_text"], "🤭 けんじ、大好き〜！");

    // 4. Simulate server restart: new AppState loads persisted config.toml automatically
    let restarted_state = AppState::new(config);
    let reloaded = restarted_state.current_settings();
    assert_eq!(reloaded.raw.general.default_engine, "irodori");
    assert!(reloaded
        .raw
        .dictionary
        .iter()
        .any(|d| d.surface == "賢嗣" && d.reading == "けんじ"));

    let _ = std::fs::remove_file(&tmp_toml);
}

#[tokio::test]
async fn test_carry_tag_propagation_across_all_three_engines_preview() {
    let app = create_router(Arc::new(AppState::new(Config::default())));
    let multi_sentence_input =
        "[mood:gently] [whisper] ねえパパ。 [laughter] ふふっ！ ずっと一緒だよ。 [calm] さて本題だよ。";

    // 1. Zonos 2: 4 chunks, accurate_mode=false on all, Carry [whisper] persists to chunk 2, [calm] resets chunk 3
    let z2_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/preview")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "zonos2",
                        "input": multi_sentence_input
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(z2_resp.status(), StatusCode::OK);
    let z2_json: Value =
        serde_json::from_slice(&z2_resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let z2_chunks = z2_json["parameters"]["chunk_parameters"].as_array().unwrap();
    assert_eq!(z2_chunks.len(), 4);
    assert_eq!(z2_chunks[0]["emotion_enabled"], true);
    assert_eq!(z2_chunks[0]["accurate_mode"], false);
    assert_eq!(z2_chunks[1]["emotion_enabled"], true);
    assert_eq!(z2_chunks[2]["emotion_enabled"], true);
    // Chunk 2 ("ずっと一緒だよ。") inherits Carry [whisper] + BaseMood [gently] without [laughter]
    assert_eq!(
        z2_chunks[2]["emotion_sliders"]["happy"],
        z2_chunks[0]["emotion_sliders"]["happy"]
    );
    // Chunk 3 ("[calm] さて本題だよ。") resets to calm while keeping accurate_mode=false!
    assert_eq!(z2_chunks[3]["emotion_enabled"], false);
    assert_eq!(z2_chunks[3]["accurate_mode"], false);

    // 2. Zonos v0.1: 4 chunks, Carry [whisper] lowers speaking_rate on chunks 0..=2, [calm] restores default 15.0 on chunk 3
    let z1_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/preview")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "zonos",
                        "input": multi_sentence_input
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(z1_resp.status(), StatusCode::OK);
    let z1_json: Value =
        serde_json::from_slice(&z1_resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let z1_chunks = z1_json["parameters"]["chunk_parameters"].as_array().unwrap();
    assert_eq!(z1_chunks.len(), 4);
    assert!(z1_chunks[0]["speaking_rate"].as_f64().unwrap() < 15.0);
    assert!(z1_chunks[2]["speaking_rate"].as_f64().unwrap() < 15.0);
    // Chunk 2 does NOT have chunk 1's [laughter] happiness spike!
    assert!(
        z1_chunks[1]["happiness"].as_f64().unwrap() > z1_chunks[2]["happiness"].as_f64().unwrap()
    );
    // Chunk 3 ([calm]) returns to baseline speaking_rate = 15.0
    assert_eq!(z1_chunks[3]["speaking_rate"].as_f64().unwrap(), 15.0);

    // 3. Irodori: Carry 👂 propagates across sentences 0..=2, [laughter] adds 😆 on sentence 1, [calm] strips cleanly on sentence 3
    let iro_resp = app
        .oneshot(
            Request::builder()
                .uri("/api/preview")
                .method("POST")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "model": "irodori-tts",
                        "input": multi_sentence_input
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(iro_resp.status(), StatusCode::OK);
    let iro_json: Value =
        serde_json::from_slice(&iro_resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(
        iro_json["processed_text"],
        "👂 ねえパパ。 👂 😆 ふふっ！ 👂 ずっと一緒だよ。 さて本題だよ。"
    );
}
