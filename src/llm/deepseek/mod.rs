//! DeepSeek Responses API client (`POST /responses`).
//!
//! Stateless: the server does not store conversation history. Multi-turn
//! callers must resend the full `input` on every request; [`Response::to_input_items`]
//! converts a previous [`Response`] into those items.

pub mod request;
pub mod response;

use std::fmt;
use std::time::Duration;

use request::CreateResponseRequest;
use reqwest::header::HeaderMap;
use reqwest::{Client, Url};
use response::{Response, ResponseFailure};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
/// Long enough for thinking mode plus a large `max_output_tokens`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Error bodies are logged and rendered with `Display`; keep them short.
const MAX_ERROR_BODY_CHARS: usize = 512;

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
    base_url: Url,
}

impl fmt::Debug for DeepSeekClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeepSeekClient")
            .field("api_key", &REDACTED)
            .field("base_url", &self.base_url.as_str())
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
    /// The request is validated locally first; `request.stream == Some(true)`
    /// is rejected because streaming is not implemented yet.
    ///
    /// Note that DeepSeek reports generation failures with HTTP 200 and
    /// `status: "failed"`, so a successful return only means the HTTP call
    /// succeeded. Use [`DeepSeekClient::create_checked`] or
    /// [`Response::ensure_success`] when that distinction matters.
    ///
    /// No automatic retry or backoff is performed: 429 and 5xx responses
    /// surface as [`Error::Api`] with `retry_after` and `request_id` populated
    /// when the server sends those headers, so callers decide the policy.
    pub async fn create(&self, request: &CreateResponseRequest) -> Result<Response, Error> {
        request.validate().map_err(Error::InvalidRequest)?;
        if request.stream == Some(true) {
            return Err(Error::StreamingNotImplemented);
        }

        let url = self.endpoint()?;
        let http_response = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await?;

        let status = http_response.status();
        let headers = http_response.headers().clone();
        let body = http_response.text().await?;

        if !status.is_success() {
            return Err(Error::Api {
                status: status.as_u16(),
                retry_after: header_string(&headers, &["retry-after"]),
                request_id: header_string(
                    &headers,
                    &["x-request-id", "request-id", "x-ds-request-id"],
                ),
                body,
            });
        }

        serde_json::from_str(&body).map_err(|source| Error::Decode { source, body })
    }

    /// Like [`DeepSeekClient::create`], but also requires `status: "completed"`.
    pub async fn create_checked(&self, request: &CreateResponseRequest) -> Result<Response, Error> {
        let response = self.create(request).await?;
        response.ensure_success().map_err(Error::ResponseFailed)?;
        Ok(response)
    }

    fn endpoint(&self) -> Result<Url, Error> {
        self.base_url
            .join("responses")
            .map_err(|source| Error::InvalidBaseUrl {
                value: self.base_url.to_string(),
                reason: source.to_string(),
            })
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
    connect_timeout: Duration,
    allow_insecure_http: bool,
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
            .field("connect_timeout", &self.connect_timeout)
            .field("allow_insecure_http", &self.allow_insecure_http)
            .finish()
    }
}

impl Default for DeepSeekClientBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_owned(),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            allow_insecure_http: false,
        }
    }
}

impl DeepSeekClientBuilder {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Absolute URL of the API root, for example `https://api.deepseek.com`.
    ///
    /// A trailing slash is added when missing, so a proxy prefix such as
    /// `https://gateway.example/v1` becomes `https://gateway.example/v1/responses`.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Total timeout for one request, including the response body.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn connect_timeout(mut self, connect_timeout: Duration) -> Self {
        self.connect_timeout = connect_timeout;
        self
    }

    /// Allow a plain `http:` base URL (local test servers only).
    ///
    /// The default is `false`: sending a bearer token over cleartext is
    /// rejected at build time.
    pub fn allow_insecure_http(mut self, allow: bool) -> Self {
        self.allow_insecure_http = allow;
        self
    }

    pub fn build(self) -> Result<DeepSeekClient, Error> {
        let api_key = self.api_key.ok_or(Error::MissingApiKey)?;
        let api_key = api_key.trim().to_owned();
        if api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }

        let mut base_url =
            Url::parse(self.base_url.trim()).map_err(|source| Error::InvalidBaseUrl {
                value: self.base_url.clone(),
                reason: source.to_string(),
            })?;

        match base_url.scheme() {
            "https" => {}
            "http" if self.allow_insecure_http => {}
            "http" => {
                return Err(Error::InvalidBaseUrl {
                    value: self.base_url,
                    reason:
                        "plain http is refused because it would send the API key in cleartext; \
                             call `allow_insecure_http(true)` for local testing"
                            .to_owned(),
                });
            }
            scheme => {
                return Err(Error::InvalidBaseUrl {
                    value: self.base_url,
                    reason: format!("unsupported scheme `{scheme}`"),
                });
            }
        }

        if base_url.query().is_some() || base_url.fragment().is_some() {
            return Err(Error::InvalidBaseUrl {
                value: self.base_url,
                reason: "the base URL must not contain a query string or fragment".to_owned(),
            });
        }

        if !base_url.path().ends_with('/') {
            let path = format!("{}/", base_url.path());
            base_url.set_path(&path);
        }

        let http = Client::builder()
            .timeout(self.timeout)
            .connect_timeout(self.connect_timeout)
            .https_only(!self.allow_insecure_http)
            .build()?;

        Ok(DeepSeekClient {
            http,
            api_key,
            base_url,
        })
    }
}

fn header_string(headers: &HeaderMap, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        headers
            .get(*name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    })
}

fn preview_body(body: &str) -> String {
    let mut chars = body.chars();
    let preview: String = chars.by_ref().take(MAX_ERROR_BODY_CHARS).collect();
    if chars.next().is_some() {
        format!("{preview} … ({} bytes total, truncated)", body.len())
    } else {
        preview
    }
}

#[derive(Debug)]
pub enum Error {
    MissingApiKey,
    InvalidBaseUrl {
        value: String,
        reason: String,
    },
    StreamingNotImplemented,
    InvalidRequest(String),
    Http(reqwest::Error),
    Api {
        status: u16,
        retry_after: Option<String>,
        request_id: Option<String>,
        body: String,
    },
    Decode {
        source: serde_json::Error,
        body: String,
    },
    ResponseFailed(Box<ResponseFailure>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingApiKey => write!(f, "DeepSeek API key is missing"),
            Self::InvalidBaseUrl { value, reason } => {
                write!(f, "invalid DeepSeek base URL `{value}`: {reason}")
            }
            Self::StreamingNotImplemented => {
                write!(f, "streaming responses are not implemented yet")
            }
            Self::InvalidRequest(message) => {
                write!(f, "invalid DeepSeek request: {message}")
            }
            Self::Http(err) => write!(f, "HTTP error: {err}"),
            Self::Api {
                status,
                retry_after,
                request_id,
                body,
            } => {
                write!(f, "DeepSeek API error ({status})")?;
                if let Some(request_id) = request_id {
                    write!(f, " [request id: {request_id}]")?;
                }
                if let Some(retry_after) = retry_after {
                    write!(f, " [retry after: {retry_after}]")?;
                }
                write!(f, ": {}", preview_body(body))
            }
            Self::Decode { source, .. } => {
                write!(f, "failed to decode DeepSeek response: {source}")
            }
            Self::ResponseFailed(failure) => write!(f, "{failure}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Http(err) => Some(err),
            Self::Decode { source, .. } => Some(source),
            Self::ResponseFailed(failure) => Some(failure.as_ref()),
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
    use axum::extract::State;
    use axum::http::{HeaderMap as AxumHeaderMap, HeaderValue, StatusCode, header};
    use axum::response::{IntoResponse, Response as AxumResponse};
    use axum::routing::post;
    use axum::{Json as AxumJson, Router};
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

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

    #[test]
    fn rejects_blank_api_key() {
        let err = DeepSeekClient::builder()
            .api_key("   ")
            .build()
            .expect_err("blank key must be rejected");

        assert!(matches!(err, Error::MissingApiKey));
    }

    #[test]
    fn rejects_relative_base_url() {
        let err = DeepSeekClient::builder()
            .api_key("sk-test")
            .base_url("api.deepseek.com")
            .build()
            .expect_err("relative base URL must be rejected");

        assert!(matches!(err, Error::InvalidBaseUrl { .. }));
    }

    #[test]
    fn rejects_plain_http_base_url_by_default() {
        let err = DeepSeekClient::builder()
            .api_key("sk-test")
            .base_url("http://127.0.0.1:1234")
            .build()
            .expect_err("http must be rejected by default");

        assert!(matches!(err, Error::InvalidBaseUrl { .. }));
        assert!(err.to_string().contains("cleartext"));
    }

    #[test]
    fn truncates_large_api_body_in_display() {
        let err = Error::Api {
            status: 502,
            retry_after: Some("7".to_owned()),
            request_id: Some("req_1".to_owned()),
            body: "x".repeat(4000),
        };
        let rendered = err.to_string();

        assert!(rendered.contains("request id: req_1"));
        assert!(rendered.contains("retry after: 7"));
        assert!(rendered.contains("truncated"));
        assert!(rendered.len() < 2000, "body was not truncated: {rendered}");
    }

    #[derive(Clone, Default)]
    struct Capture {
        headers: Arc<Mutex<Option<AxumHeaderMap>>>,
        body: Arc<Mutex<Option<Value>>>,
    }

    struct TestServer {
        base_url: String,
        handle: tokio::task::JoinHandle<()>,
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.handle.abort();
        }
    }

    async fn spawn(app: Router) -> TestServer {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        TestServer {
            base_url: format!("http://{addr}/"),
            handle,
        }
    }

    fn completed_response() -> Value {
        json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "deepseek-flash",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{ "type": "output_text", "text": "hello", "annotations": [], "logprobs": [] }]
            }],
            "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 }
        })
    }

    async fn capture_handler(
        State(capture): State<Capture>,
        headers: AxumHeaderMap,
        AxumJson(body): AxumJson<Value>,
    ) -> AxumJson<Value> {
        *capture.headers.lock().unwrap() = Some(headers);
        *capture.body.lock().unwrap() = Some(body);
        AxumJson(completed_response())
    }

    async fn rate_limited_handler() -> AxumResponse {
        let mut response = (StatusCode::TOO_MANY_REQUESTS, "slow down").into_response();
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("7"));
        response
            .headers_mut()
            .insert("x-request-id", HeaderValue::from_static("req_123"));
        response
    }

    async fn not_json_handler() -> AxumResponse {
        (StatusCode::OK, "<html>oops</html>").into_response()
    }

    async fn failed_handler() -> AxumJson<Value> {
        AxumJson(json!({
            "id": "resp_failed",
            "object": "response",
            "created_at": 1,
            "status": "failed",
            "model": "deepseek-flash",
            "output": [],
            "error": { "code": "server_error", "message": "boom" }
        }))
    }

    fn insecure_client(server: &TestServer) -> DeepSeekClient {
        DeepSeekClient::builder()
            .api_key(SENTINEL_KEY)
            .base_url(format!("{}v1/", server.base_url))
            .allow_insecure_http(true)
            .build()
            .expect("client should build")
    }

    #[tokio::test]
    async fn posts_to_responses_with_bearer_auth() {
        let capture = Capture::default();
        let app = Router::new()
            .route("/v1/responses", post(capture_handler))
            .with_state(capture.clone());
        let server = spawn(app).await;
        let client = insecure_client(&server);

        let request = CreateResponseRequest::new("deepseek-flash").input_text("hi");
        let response = client.create(&request).await.expect("request succeeds");

        assert_eq!(response.output_text(), "hello");

        let headers = capture.headers.lock().unwrap().clone().unwrap();
        assert_eq!(
            headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some(format!("Bearer {SENTINEL_KEY}").as_str())
        );
        assert_eq!(
            headers
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );

        let body = capture.body.lock().unwrap().clone().unwrap();
        assert_eq!(body["input"], "hi");
        assert!(body.get("stream").is_none());
    }

    #[tokio::test]
    async fn surfaces_retry_after_and_request_id() {
        let app = Router::new().route("/responses", post(rate_limited_handler));
        let server = spawn(app).await;
        let client = DeepSeekClient::builder()
            .api_key(SENTINEL_KEY)
            .base_url(server.base_url.clone())
            .allow_insecure_http(true)
            .build()
            .unwrap();

        let request = CreateResponseRequest::new("deepseek-flash").input_text("hi");
        let err = client.create(&request).await.unwrap_err();

        match err {
            Error::Api {
                status,
                retry_after,
                request_id,
                body,
            } => {
                assert_eq!(status, 429);
                assert_eq!(retry_after.as_deref(), Some("7"));
                assert_eq!(request_id.as_deref(), Some("req_123"));
                assert!(body.contains("slow down"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn maps_non_json_success_to_decode_error() {
        let app = Router::new().route("/responses", post(not_json_handler));
        let server = spawn(app).await;
        let client = DeepSeekClient::builder()
            .api_key(SENTINEL_KEY)
            .base_url(server.base_url.clone())
            .allow_insecure_http(true)
            .build()
            .unwrap();

        let request = CreateResponseRequest::new("deepseek-flash").input_text("hi");
        let err = client.create(&request).await.unwrap_err();

        match err {
            Error::Decode { source, body } => {
                assert!(source.is_syntax());
                assert!(body.contains("<html>"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn create_checked_rejects_failed_response() {
        let app = Router::new().route("/responses", post(failed_handler));
        let server = spawn(app).await;
        let client = DeepSeekClient::builder()
            .api_key(SENTINEL_KEY)
            .base_url(server.base_url.clone())
            .allow_insecure_http(true)
            .build()
            .unwrap();

        let request = CreateResponseRequest::new("deepseek-flash").input_text("hi");
        let err = client.create_checked(&request).await.unwrap_err();

        assert!(matches!(err, Error::ResponseFailed(_)));
        assert!(err.to_string().contains("server_error"));
    }

    #[tokio::test]
    async fn create_rejects_streaming() {
        let client = DeepSeekClient::new("sk-test").unwrap();
        let request = CreateResponseRequest::new("deepseek-flash")
            .input_text("hi")
            .stream(true);

        let err = client.create(&request).await.unwrap_err();

        assert!(matches!(err, Error::StreamingNotImplemented));
    }

    #[tokio::test]
    async fn rejects_invalid_request_before_sending() {
        let client = DeepSeekClient::new("sk-test").unwrap();
        let request = CreateResponseRequest::new("deepseek-flash");

        let err = client.create(&request).await.unwrap_err();

        assert!(matches!(err, Error::InvalidRequest(_)));
        assert!(err.to_string().contains("at least one"));
    }
}
