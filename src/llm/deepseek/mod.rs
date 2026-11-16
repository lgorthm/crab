//! DeepSeek Responses API client (`POST /responses`).
//!
//! Stateless: the server does not store conversation history. Multi-turn
//! callers must resend the full `input` on every request.

pub mod request;
pub mod response;

use request::CreateResponseRequest;
use reqwest::Client;
use response::Response;
use std::fmt;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Printed instead of a secret value by every `Debug` implementation below.
const REDACTED: &str = "<redacted>";
/// Printed for a builder that does not hold a key yet.
const UNSET: &str = "<unset>";

/// DeepSeek Responses API client (`POST /responses`).
///
/// The API key is a secret: this type's `Debug` implementation is written by
/// hand and never prints it. Do not replace that impl with `#[derive(Debug)]`,
/// and do not add another field carrying key material.
#[derive(Clone)]
pub struct DeepSeekClient {
    http: Client,
    api_key: String,
    base_url: String,
}

impl fmt::Debug for DeepSeekClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeepSeekClient")
            .field("api_key", &REDACTED)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl DeepSeekClient {
    pub fn new(api_key: impl Into<String>) -> Result<Self, Error> {
        Self::builder().api_key(api_key).build()
    }

    pub fn builder() -> DeepSeekClientBuilder {
        DeepSeekClientBuilder::default()
    }

    /// Create a model response (`POST /responses`, non-streaming).
    ///
    /// Set `request.stream = Some(true)` is rejected here; use a streaming
    /// helper when that is implemented.
    pub async fn create(&self, mut request: CreateResponseRequest) -> Result<Response, Error> {
        if request.stream == Some(true) {
            return Err(Error::StreamingNotImplemented);
        }
        // Force non-streaming for this method.
        request.stream = Some(false);

        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let http_response = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        let status = http_response.status();
        let body = http_response.text().await?;

        if !status.is_success() {
            return Err(Error::Api {
                status: status.as_u16(),
                body,
            });
        }

        serde_json::from_str(&body).map_err(|source| Error::Decode { source, body })
    }
}

/// Builder for [`DeepSeekClient`].
///
/// Like the client, the builder's `Debug` implementation is hand written so a
/// configured key is only ever rendered as a placeholder.
pub struct DeepSeekClientBuilder {
    api_key: Option<String>,
    base_url: String,
    timeout: Duration,
}

impl fmt::Debug for DeepSeekClientBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let api_key = if self.api_key.is_some() {
            REDACTED
        } else {
            UNSET
        };
        f.debug_struct("DeepSeekClientBuilder")
            .field("api_key", &api_key)
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl Default for DeepSeekClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }
}

impl DeepSeekClientBuilder {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn build(self) -> Result<DeepSeekClient, Error> {
        let api_key = self.api_key.ok_or(Error::MissingApiKey)?;
        if api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }

        let http = Client::builder().timeout(self.timeout).build()?;

        Ok(DeepSeekClient {
            http,
            api_key,
            base_url: self.base_url,
        })
    }
}

#[derive(Debug)]
pub enum Error {
    MissingApiKey,
    StreamingNotImplemented,
    Http(reqwest::Error),
    Api {
        status: u16,
        body: String,
    },
    Decode {
        source: serde_json::Error,
        body: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingApiKey => write!(f, "DeepSeek API key is missing"),
            Self::StreamingNotImplemented => {
                write!(f, "streaming responses are not implemented yet")
            }
            Self::Http(err) => write!(f, "HTTP error: {err}"),
            Self::Api { status, body } => write!(f, "DeepSeek API error ({status}): {body}"),
            Self::Decode { source, .. } => write!(f, "failed to decode response: {source}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Http(err) => Some(err),
            Self::Decode { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SENTINEL_KEY: &str = "sk-deadbeef0123456789-this-must-not-be-logged";

    #[test]
    fn client_debug_redacts_api_key() {
        let client = DeepSeekClient::new(SENTINEL_KEY).expect("client should build");
        let rendered = format!("{client:?}");

        assert!(
            !rendered.contains(SENTINEL_KEY),
            "Debug output leaked the API key: {rendered}"
        );
        assert!(
            rendered.contains(REDACTED),
            "expected redaction marker in {rendered}"
        );
        assert!(
            rendered.contains(DEFAULT_BASE_URL),
            "expected base URL in {rendered}"
        );
    }

    #[test]
    fn cloned_client_debug_redacts_api_key() {
        let client = DeepSeekClient::new(SENTINEL_KEY)
            .expect("client should build")
            .clone();
        let rendered = format!("{client:?}");

        assert!(!rendered.contains(SENTINEL_KEY), "clone leaked the API key");
    }

    #[test]
    fn builder_debug_redacts_api_key() {
        let builder = DeepSeekClient::builder()
            .api_key(SENTINEL_KEY)
            .base_url("https://example.invalid");
        let rendered = format!("{builder:?}");

        assert!(
            !rendered.contains(SENTINEL_KEY),
            "builder Debug output leaked the API key: {rendered}"
        );
        assert!(rendered.contains(REDACTED));
    }

    #[test]
    fn builder_debug_marks_unset_key() {
        let rendered = format!("{:?}", DeepSeekClient::builder());

        assert!(rendered.contains(UNSET));
        assert!(!rendered.contains(REDACTED));
    }
}
