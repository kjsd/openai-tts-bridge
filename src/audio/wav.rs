use byteorder::{LittleEndian, WriteBytesExt};

/// Converts raw 32-bit float PCM (e.g. 44.1kHz, Mono, Little-Endian)
/// into standard 16-bit signed integer RIFF WAVE (Format 1).
pub fn float32_to_pcm16_wav(
    raw_float_pcm: &[u8],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>, String> {
    if raw_float_pcm.len() % 4 != 0 {
        return Err(format!(
            "Input raw PCM length must be multiple of 4 bytes (Float32), got {}",
            raw_float_pcm.len()
        ));
    }

    let num_samples = raw_float_pcm.len() / 4;
    let data_len = (num_samples * 2) as u32;
    let riff_chunk_size = 36 + data_len;

    // Pre-allocate full buffer: 44 bytes header + PCM 16 data
    let mut out = Vec::with_capacity(44 + (num_samples * 2));

    // 1. RIFF header
    out.extend_from_slice(b"RIFF");
    out.write_u32::<LittleEndian>(riff_chunk_size)
        .map_err(|e| e.to_string())?;
    out.extend_from_slice(b"WAVE");

    // 2. fmt subchunk
    out.extend_from_slice(b"fmt ");
    out.write_u32::<LittleEndian>(16)
        .map_err(|e| e.to_string())?; // Subchunk1Size for PCM
    out.write_u16::<LittleEndian>(1)
        .map_err(|e| e.to_string())?; // AudioFormat: 1 = PCM (Integer)
    out.write_u16::<LittleEndian>(channels)
        .map_err(|e| e.to_string())?;
    out.write_u32::<LittleEndian>(sample_rate)
        .map_err(|e| e.to_string())?;

    let byte_rate = sample_rate * (channels as u32) * 2;
    let block_align = channels * 2;
    out.write_u32::<LittleEndian>(byte_rate)
        .map_err(|e| e.to_string())?;
    out.write_u16::<LittleEndian>(block_align)
        .map_err(|e| e.to_string())?;
    out.write_u16::<LittleEndian>(16)
        .map_err(|e| e.to_string())?; // BitsPerSample: 16

    // 3. data subchunk
    out.extend_from_slice(b"data");
    out.write_u32::<LittleEndian>(data_len)
        .map_err(|e| e.to_string())?;

    // 4. Convert float32 samples to int16 samples
    for chunk in raw_float_pcm.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);

        let sample: i16 = if f.is_nan() || f.is_infinite() {
            0
        } else if f >= 1.0 {
            32767
        } else if f <= -1.0 {
            -32768
        } else if f >= 0.0 {
            (f * 32767.0).round() as i16
        } else {
            (f * 32768.0).round() as i16
        };

        out.write_i16::<LittleEndian>(sample)
            .map_err(|e| e.to_string())?;
    }

    Ok(out)
}

#[derive(Debug, Clone)]
struct ParsedWav {
    fmt_body: Vec<u8>,
    data_body: Vec<u8>,
}

fn parse_wav_chunks(wav: &[u8]) -> Result<ParsedWav, String> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err("Invalid RIFF WAVE header".to_string());
    }

    let mut pos = 12;
    let mut fmt_body: Option<Vec<u8>> = None;
    let mut data_parts: Vec<&[u8]> = Vec::new();

    while pos + 8 <= wav.len() {
        let chunk_id = &wav[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            wav[pos + 4],
            wav[pos + 5],
            wav[pos + 6],
            wav[pos + 7],
        ]) as usize;
        pos += 8;

        let end = (pos + chunk_size).min(wav.len());
        let body = &wav[pos..end];

        match chunk_id {
            b"fmt " => {
                if fmt_body.is_none() {
                    fmt_body = Some(body.to_vec());
                }
            }
            b"data" => {
                data_parts.push(body);
            }
            _ => {}
        }

        pos = end;
        // RIFF chunks are word-aligned (padded to even byte boundary)
        if chunk_size % 2 == 1 && pos < wav.len() {
            pos += 1;
        }
    }

    let fmt = fmt_body.ok_or_else(|| "Missing 'fmt ' chunk in WAV".to_string())?;
    if fmt.len() < 16 {
        return Err("Incomplete 'fmt ' chunk in WAV".to_string());
    }
    if data_parts.is_empty() {
        return Err("Missing 'data' chunk in WAV".to_string());
    }

    let total_data: usize = data_parts.iter().map(|p| p.len()).sum();
    let mut raw_data = Vec::with_capacity(total_data);
    for p in data_parts {
        raw_data.extend_from_slice(p);
    }

    // Normalize IEEE Float32 WAV (format 3, 32-bit) into canonical PCM16 WAV (format 1, 16-bit)
    let audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
    let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
    let sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    let bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);

    if audio_format == 3 && bits_per_sample == 32 {
        let converted = float32_to_pcm16_wav(&raw_data, sample_rate, channels)?;
        return parse_wav_chunks(&converted);
    }

    Ok(ParsedWav {
        fmt_body: fmt,
        data_body: raw_data,
    })
}

fn build_silence_bytes(fmt_body: &[u8], silence_ms: u32) -> Vec<u8> {
    if fmt_body.len() < 14 || silence_ms == 0 {
        return Vec::new();
    }
    let byte_rate = u32::from_le_bytes([fmt_body[8], fmt_body[9], fmt_body[10], fmt_body[11]]);
    let block_align = u16::from_le_bytes([fmt_body[12], fmt_body[13]]) as usize;
    if byte_rate == 0 || block_align == 0 {
        return Vec::new();
    }

    let raw_bytes = ((byte_rate as u64) * (silence_ms as u64) / 1000) as usize;
    let frames = raw_bytes / block_align;
    vec![0u8; frames * block_align]
}

fn build_canonical_wav(fmt_body: &[u8], data_body: &[u8]) -> Result<Vec<u8>, String> {
    let fmt_len = fmt_body.len() as u32;
    let data_len = data_body.len() as u32;
    let riff_size = 4 + (8 + fmt_len) + (8 + data_len);

    let mut out = Vec::with_capacity((8 + riff_size) as usize);
    out.extend_from_slice(b"RIFF");
    out.write_u32::<LittleEndian>(riff_size)
        .map_err(|e| e.to_string())?;
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.write_u32::<LittleEndian>(fmt_len)
        .map_err(|e| e.to_string())?;
    out.extend_from_slice(fmt_body);
    out.extend_from_slice(b"data");
    out.write_u32::<LittleEndian>(data_len)
        .map_err(|e| e.to_string())?;
    out.extend_from_slice(data_body);
    Ok(out)
}

/// Concatenates one or more RIFF WAVE binaries into a single canonical PCM RIFF WAVE binary,
/// inserting `silence_ms` milliseconds of digital silence between chunks.
pub fn concat_wav_chunks(wav_chunks: &[Vec<u8>], silence_ms: u32) -> Result<Vec<u8>, String> {
    if wav_chunks.is_empty() {
        return Err("No WAV chunks provided for concatenation".to_string());
    }

    let mut parsed_list = Vec::with_capacity(wav_chunks.len());
    for chunk in wav_chunks {
        parsed_list.push(parse_wav_chunks(chunk)?);
    }

    let first_fmt = &parsed_list[0].fmt_body;
    if parsed_list.len() == 1 {
        return build_canonical_wav(first_fmt, &parsed_list[0].data_body);
    }

    let silence = build_silence_bytes(first_fmt, silence_ms);
    let total_data_len: usize = parsed_list.iter().map(|p| p.data_body.len()).sum::<usize>()
        + silence.len() * (parsed_list.len() - 1);

    let mut combined = Vec::with_capacity(total_data_len);
    for (idx, p) in parsed_list.iter().enumerate() {
        if idx > 0 && !silence.is_empty() {
            combined.extend_from_slice(&silence);
        }
        combined.extend_from_slice(&p.data_body);
    }

    build_canonical_wav(first_fmt, &combined)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float32_to_pcm16_wav_header() {
        let mut input = Vec::new();
        input.write_f32::<LittleEndian>(0.0).unwrap();
        input.write_f32::<LittleEndian>(0.5).unwrap();

        let wav = float32_to_pcm16_wav(&input, 44100, 1).expect("Conversion failed");

        assert_eq!(wav.len(), 48);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");

        let format_code = u16::from_le_bytes([wav[20], wav[21]]);
        assert_eq!(format_code, 1);

        let channels = u16::from_le_bytes([wav[22], wav[23]]);
        assert_eq!(channels, 1);

        let sample_rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        assert_eq!(sample_rate, 44100);

        let bits_per_sample = u16::from_le_bytes([wav[34], wav[35]]);
        assert_eq!(bits_per_sample, 16);

        let s0 = i16::from_le_bytes([wav[44], wav[45]]);
        let s1 = i16::from_le_bytes([wav[46], wav[47]]);
        assert_eq!(s0, 0);
        assert!((s1 - 16384).abs() <= 1);
    }

    #[test]
    fn test_clipping_protection() {
        let mut input = Vec::new();
        input.write_f32::<LittleEndian>(1.5).unwrap();
        input.write_f32::<LittleEndian>(-2.0).unwrap();
        input.write_f32::<LittleEndian>(f32::NAN).unwrap();

        let wav = float32_to_pcm16_wav(&input, 44100, 1).expect("Conversion failed");
        assert_eq!(wav.len(), 44 + 6);

        let s0 = i16::from_le_bytes([wav[44], wav[45]]);
        let s1 = i16::from_le_bytes([wav[46], wav[47]]);
        let s2 = i16::from_le_bytes([wav[48], wav[49]]);

        assert_eq!(s0, 32767);
        assert_eq!(s1, -32768);
        assert_eq!(s2, 0);
    }

    #[test]
    fn test_concat_wav_chunks_with_silence() {
        let mut pcm1 = Vec::new();
        pcm1.write_f32::<LittleEndian>(0.5).unwrap();
        let wav1 = float32_to_pcm16_wav(&pcm1, 1000, 1).unwrap(); // 1000 Hz, 2 bytes/sample

        let mut pcm2 = Vec::new();
        pcm2.write_f32::<LittleEndian>(-0.5).unwrap();
        let wav2 = float32_to_pcm16_wav(&pcm2, 1000, 1).unwrap();

        // 100ms at 1000Hz mono 16-bit = 100 frames = 200 bytes of silence
        let merged = concat_wav_chunks(&[wav1, wav2], 100).expect("Concat failed");
        // 44 header + 2 (chunk1) + 200 (silence) + 2 (chunk2) = 248 bytes
        assert_eq!(merged.len(), 248);
        assert_eq!(&merged[0..4], b"RIFF");
        let data_size = u32::from_le_bytes([merged[40], merged[41], merged[42], merged[43]]);
        assert_eq!(data_size, 204);
    }
}
