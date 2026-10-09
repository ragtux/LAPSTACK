// SPDX-FileCopyrightText: 2026 RAGTUX LLC
// SPDX-License-Identifier: MIT

//! Content credentials (C2PA, https://contentcredentials.org) for the browser
//! app: a self-signed ES256 certificate made in the browser, and a signed
//! manifest embedded into a PNG / JPEG / GIF. Verifiers show the signer as
//! unknown (the certificate is not on any trust list), but the manifest,
//! its assertions and the content hash validate. Built as its own wasm module
//! (web/pkg-cc) because c2pa is large; worker.js imports it on first use.
use std::io::Cursor;
use std::str::FromStr;
use std::time::Duration;

use c2pa::{Builder, create_signer, crypto::raw_signature::SigningAlg};
use p256::ecdsa::{DerSignature, SigningKey};
use p256::pkcs8::EncodePrivateKey;
use wasm_bindgen::prelude::*;
use x509_cert::builder::{Builder as _, CertificateBuilder, Profile};
use x509_cert::der::asn1::{GeneralizedTime, UtcTime};
use x509_cert::der::pem::LineEnding;
use x509_cert::der::EncodePem;
use x509_cert::ext::pkix::ExtendedKeyUsage;
use x509_cert::name::Name;
use x509_cert::serial_number::SerialNumber;
use x509_cert::spki::SubjectPublicKeyInfoOwned;
use x509_cert::time::{Time, Validity};

fn js<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// A self-signed ES256 certificate for `name`, valid ten years from `now_secs`
/// (unix time, passed in: wasm has no clock): `(cert_pem, key_pem)`.
pub fn make_cert_pem(name: &str, now_secs: u64) -> Result<(String, String), String> {
    let e = |x: &dyn std::fmt::Display| x.to_string();
    let key = SigningKey::random(&mut rand_core::OsRng);
    let spki = SubjectPublicKeyInfoOwned::from_key(*key.verifying_key()).map_err(|x| e(&x))?;
    // the C2PA validator reads the signer's name from the Organization attribute
    let n: String = name
        .chars()
        .map(|c| if matches!(c, ',' | '=' | '+' | '"' | '\\' | '<' | '>' | ';' | '#') { ' ' } else { c })
        .collect();
    let n = n.trim();
    let n = if n.is_empty() { "lapstack user" } else { n };
    let subject = Name::from_str(&format!("CN={n},O={n}")).map_err(|x| e(&x))?;
    let not_before = Time::UtcTime(UtcTime::from_unix_duration(Duration::from_secs(now_secs.saturating_sub(600))).map_err(|x| e(&x))?);
    let not_after = Time::GeneralTime(GeneralizedTime::from_unix_duration(Duration::from_secs(now_secs + 10 * 365 * 86400)).map_err(|x| e(&x))?);
    let validity = Validity { not_before, not_after };
    let serial = SerialNumber::from(now_secs as u32);
    let profile = Profile::Leaf { issuer: subject.clone(), enable_key_agreement: false, enable_key_encipherment: false };
    let mut b = CertificateBuilder::new(profile, serial, validity, subject, spki, &key).map_err(|x| e(&x))?;
    b.add_extension(&ExtendedKeyUsage(vec![const_oid::db::rfc5280::ID_KP_EMAIL_PROTECTION])).map_err(|x| e(&x))?;
    let cert = b.build::<DerSignature>().map_err(|x| e(&x))?;
    let cert_pem = cert.to_pem(LineEnding::LF).map_err(|x| e(&x))?;
    let key_pem = key.to_pkcs8_pem(LineEnding::LF).map_err(|x| e(&x))?.to_string();
    Ok((cert_pem, key_pem))
}

/// `[cert_pem, key_pem]`, see `make_cert_pem`.
#[wasm_bindgen]
pub fn make_cert(name: &str, now_secs: f64) -> Result<Vec<JsValue>, JsValue> {
    let (c, k) = make_cert_pem(name, now_secs.max(0.0) as u64).map_err(js)?;
    Ok(vec![JsValue::from_str(&c), JsValue::from_str(&k)])
}

/// Embed a manifest (c2pa Builder JSON) into an image of MIME type `mime`,
/// signed with the certificate and key from `make_cert`; returns the new file.
pub fn sign_bytes(bytes: &[u8], mime: &str, manifest_json: &str, cert_pem: &str, key_pem: &str) -> Result<Vec<u8>, String> {
    let signer = create_signer::from_keys(cert_pem.as_bytes(), key_pem.as_bytes(), SigningAlg::Es256, None).map_err(|x| x.to_string())?;
    #[allow(deprecated)]
    let mut builder = Builder::from_json(manifest_json).map_err(|x| x.to_string())?;
    let mut src = Cursor::new(bytes);
    let mut dst = Cursor::new(Vec::with_capacity(bytes.len() + 64 * 1024));
    builder.sign(&*signer, mime, &mut src, &mut dst).map_err(|x| x.to_string())?;
    Ok(dst.into_inner())
}

#[wasm_bindgen]
pub fn sign_image(bytes: &[u8], mime: &str, manifest_json: &str, cert_pem: &str, key_pem: &str) -> Result<Vec<u8>, JsValue> {
    sign_bytes(bytes, mime, manifest_json, cert_pem, key_pem).map_err(js)
}

#[cfg(test)]
mod tests {
    #[test]
    fn sign_and_read_back() {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let (c, k) = super::make_cert_pem("lapstack test", now).unwrap();
        // a 1x1 RGB PNG
        let png: Vec<u8> = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0x0D, 0x49, 0x48, 0x44, 0x52, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0, 0x90, 0x77, 0x53, 0xDE, 0, 0, 0,
            0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0, 0, 3, 1, 1, 0, 0x18, 0xDD, 0x8D, 0xB0, 0, 0, 0, 0, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let manifest = r#"{"claim_generator_info":[{"name":"lapstack","version":"0.1.0"}],"title":"test.png","assertions":[{"label":"c2pa.actions","data":{"actions":[{"action":"c2pa.created","digitalSourceType":"http://cv.iptc.org/newscodes/digitalsourcetype/compositeCapture"}]}}]}"#;
        let out = super::sign_bytes(&png, "image/png", manifest, &c, &k).unwrap();
        assert!(out.len() > png.len());
        #[allow(deprecated)]
        let reader = c2pa::Reader::from_stream("image/png", std::io::Cursor::new(out)).unwrap();
        let report = reader.to_string();
        assert!(report.contains("\"claimSignature.validated\""), "{report}");
        assert!(report.contains("\"validation_state\": \"Valid\""), "{report}");
    }
}
