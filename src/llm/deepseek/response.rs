//! Model response from `POST /responses`.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::request::{
    InputContent, MessageRole, ResponseInputContentPart, ResponseInputItem, ResponseTool,
    ToolChoice,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub status: ResponseStatus,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponseTool>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoning>,

    /// Catch-all for fields we do not model yet (store, metadata, etc.).
    #[serde(default, flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl Response {
    /// Concatenate the `output_text` of final-answer message items.
    ///
    /// Items whose `phase` is present but not `final_answer` are skipped, so
    /// intermediate commentary never leaks into the returned answer. Items
    /// without a `phase` are kept for compatibility.
    pub fn output_text(&self) -> String {
        self.output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::Message { content, phase, .. } if is_final_message(phase) => {
                    Some(content)
                }
                _ => None,
            })
            .flat_map(|parts| parts.iter())
            .filter_map(|part| match part {
                OutputContentPart::OutputText { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// The `(call_id, name, arguments)` of every complete function call.
    ///
    /// Incomplete items (missing any of the three fields) are skipped; inspect
    /// [`Response::output`] directly when that distinction matters.
    pub fn function_calls(&self) -> Vec<(&str, &str, &str)> {
        self.output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::FunctionCall {
                    call_id: Some(call_id),
                    name: Some(name),
                    arguments: Some(arguments),
                    ..
                } => Some((call_id.as_str(), name.as_str(), arguments.as_str())),
                _ => None,
            })
            .collect()
    }

    /// Every content annotation in the response (for example web-search
    /// `url_citation` annotations).
    pub fn annotations(&self) -> Vec<&Value> {
        self.output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::Message { content, .. } => Some(content),
                _ => None,
            })
            .flat_map(|parts| parts.iter())
            .filter_map(|part| match part {
                OutputContentPart::OutputText { annotations, .. } => Some(annotations.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// Build the `input` items that replay this response in a follow-up turn.
    ///
    /// Items that cannot be expressed as request input (for example
    /// `tool_search_call`, or a malformed item missing `call_id`) are skipped.
    /// Use [`TryFrom::try_from`] on [`ResponseInputItem`] to observe those
    /// failures instead.
    ///
    /// A converted `function_call` still needs its matching
    /// `function_call_output` appended before the history is sent; the API
    /// rejects unpaired calls.
    pub fn to_input_items(&self) -> Vec<ResponseInputItem> {
        self.output
            .iter()
            .filter_map(|item| ResponseInputItem::try_from(item).ok())
            .collect()
    }

    /// Fail unless the response reached `completed`.
    ///
    /// DeepSeek reports generation failures with HTTP 200 and
    /// `status: "failed"` (and truncation as `status: "incomplete"`), so an
    /// `Ok` from [`super::DeepSeekClient::create`] does not by itself mean the
    /// model produced an answer. `create_checked` applies this check.
    pub fn ensure_success(&self) -> Result<(), Box<ResponseFailure>> {
        if matches!(&self.status, ResponseStatus::Completed) {
            return Ok(());
        }
        Err(Box::new(ResponseFailure {
            id: self.id.clone(),
            status: self.status.clone(),
            error: self.error.clone(),
            incomplete_details: self.incomplete_details.clone(),
        }))
    }

    pub fn is_failed(&self) -> bool {
        matches!(&self.status, ResponseStatus::Failed)
    }
}

fn is_final_message(phase: &Option<String>) -> bool {
    phase.as_deref().is_none_or(|phase| phase == "final_answer")
}

/// Description of a response that did not reach `completed`.
#[derive(Debug, Clone)]
pub struct ResponseFailure {
    pub id: String,
    pub status: ResponseStatus,
    pub error: Option<ResponseError>,
    pub incomplete_details: Option<IncompleteDetails>,
}

impl fmt::Display for ResponseFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeepSeek response {} ended with status `{}`",
            self.id,
            self.status.as_str()
        )?;
        if let Some(error) = &self.error {
            if let Some(code) = &error.code {
                write!(f, ": {code}")?;
            }
            if let Some(message) = &error.message {
                write!(f, " - {message}")?;
            }
        }
        if let Some(details) = &self.incomplete_details
            && let Some(reason) = &details.reason
        {
            write!(f, ": {}", reason.as_str())?;
        }
        Ok(())
    }
}

impl std::error::Error for ResponseFailure {}

/// Response status. Unknown values are preserved instead of failing the decode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Unknown(String),
}

impl ResponseStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Failed => "failed",
            Self::Unknown(value) => value,
        }
    }
}

impl From<String> for ResponseStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "incomplete" => Self::Incomplete,
            "failed" => Self::Failed,
            _ => Self::Unknown(value),
        }
    }
}

impl From<ResponseStatus> for String {
    fn from(value: ResponseStatus) -> Self {
        value.as_str().to_owned()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<IncompleteReason>,
}

/// Why a response is incomplete. Unknown values are preserved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum IncompleteReason {
    MaxOutputTokens,
    ContentFilter,
    Unknown(String),
}

impl IncompleteReason {
    pub fn as_str(&self) -> &str {
        match self {
            Self::MaxOutputTokens => "max_output_tokens",
            Self::ContentFilter => "content_filter",
            Self::Unknown(value) => value,
        }
    }
}

impl From<String> for IncompleteReason {
    fn from(value: String) -> Self {
        match value.as_str() {
            "max_output_tokens" => Self::MaxOutputTokens,
            "content_filter" => Self::ContentFilter,
            _ => Self::Unknown(value),
        }
    }
}

impl From<IncompleteReason> for String {
    fn from(value: IncompleteReason) -> Self {
        value.as_str().to_owned()
    }
}

/// The `reasoning` block echoed in a response.
///
/// Unlike the request-side [`super::request::ReasoningConfig`], this type keeps
/// `summary` (currently always `null`) and any future fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseReasoning {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<super::request::ReasoningEffort>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<Value>,

    #[serde(default, flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        status: Option<OutputItemStatus>,
        #[serde(default)]
        role: Option<MessageRole>,
        #[serde(default)]
        content: Vec<OutputContentPart>,
        #[serde(default)]
        phase: Option<String>,
    },
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        status: Option<OutputItemStatus>,
        #[serde(default)]
        content: Vec<OutputContentPart>,
    },
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        status: Option<OutputItemStatus>,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        arguments: Option<String>,
    },
    CustomToolCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        status: Option<OutputItemStatus>,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        name: Option<String>,
        /// OpenAI-compatible custom tool payload (`apply_patch` sends `input`).
        #[serde(default)]
        input: Option<String>,
        #[serde(default)]
        arguments: Option<String>,
    },
    WebSearchCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        status: Option<OutputItemStatus>,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        kind: Option<String>,
        #[serde(default)]
        action: Option<Value>,
        #[serde(default)]
        page: Option<Value>,
        #[serde(default)]
        signature: Option<String>,
        #[serde(default)]
        error: Option<Value>,
    },
    ToolSearchCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        status: Option<OutputItemStatus>,
        #[serde(default)]
        call_id: Option<String>,
    },
    /// An output item type this crate does not model yet.
    ///
    /// Only a marker is kept: the item's payload is not preserved, so it cannot
    /// be replayed by [`Response::to_input_items`].
    #[serde(other)]
    Unknown,
}

/// Status of one output item. `failed` is a documented value for failed
/// `web_search_call` items; unknown values are preserved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum OutputItemStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Unknown(String),
}

impl OutputItemStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Failed => "failed",
            Self::Unknown(value) => value,
        }
    }
}

impl From<String> for OutputItemStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "incomplete" => Self::Incomplete,
            "failed" => Self::Failed,
            _ => Self::Unknown(value),
        }
    }
}

impl From<OutputItemStatus> for String {
    fn from(value: OutputItemStatus) -> Self {
        value.as_str().to_owned()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentPart {
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Value>,
        #[serde(default)]
        logprobs: Vec<Value>,
    },
    ReasoningText {
        text: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<ServerToolUse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
    #[serde(default)]
    pub safety_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerToolUse {
    pub tool_search: u64,
    pub tool_fetch: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_image_search: Option<u64>,
}

/// Why an output item cannot be turned back into request input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemConversionError {
    /// A field the request format requires is missing on the output item.
    MissingField(&'static str),
    /// The item type has no request-side equivalent.
    Unsupported(&'static str),
}

impl fmt::Display for ItemConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(field) => {
                write!(f, "output item is missing required field `{field}`")
            }
            Self::Unsupported(item) => write!(f, "`{item}` cannot be replayed as request input"),
        }
    }
}

impl std::error::Error for ItemConversionError {}

impl TryFrom<&ResponseOutputItem> for ResponseInputItem {
    type Error = ItemConversionError;

    fn try_from(item: &ResponseOutputItem) -> Result<Self, Self::Error> {
        match item {
            ResponseOutputItem::Message { role, content, .. } => {
                let mut parts = Vec::new();
                for part in content {
                    match part {
                        OutputContentPart::OutputText { text, .. } => {
                            parts.push(ResponseInputContentPart::output_text(text));
                        }
                        OutputContentPart::ReasoningText { text } => {
                            parts.push(ResponseInputContentPart::reasoning_text(text));
                        }
                        OutputContentPart::Unknown => {}
                    }
                }
                if parts.is_empty() {
                    return Err(ItemConversionError::Unsupported(
                        "message without replayable content",
                    ));
                }
                let role = role.unwrap_or(MessageRole::Assistant);
                Ok(ResponseInputItem::message(role, InputContent::Parts(parts)))
            }
            ResponseOutputItem::Reasoning { content, .. } => {
                let mut parts = Vec::new();
                for part in content {
                    if let OutputContentPart::ReasoningText { text } = part {
                        parts.push(ResponseInputContentPart::reasoning_text(text));
                    }
                }
                if parts.is_empty() {
                    return Err(ItemConversionError::Unsupported(
                        "reasoning without reasoning_text",
                    ));
                }
                Ok(ResponseInputItem::reasoning(parts))
            }
            ResponseOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => Ok(ResponseInputItem::function_call(
                call_id
                    .clone()
                    .ok_or(ItemConversionError::MissingField("call_id"))?,
                name.clone()
                    .ok_or(ItemConversionError::MissingField("name"))?,
                arguments
                    .clone()
                    .ok_or(ItemConversionError::MissingField("arguments"))?,
            )),
            ResponseOutputItem::CustomToolCall {
                call_id,
                name,
                input,
                arguments,
                ..
            } => {
                if input.is_none() && arguments.is_none() {
                    return Err(ItemConversionError::MissingField("input"));
                }
                Ok(ResponseInputItem::CustomToolCall {
                    call_id: call_id
                        .clone()
                        .ok_or(ItemConversionError::MissingField("call_id"))?,
                    name: name
                        .clone()
                        .ok_or(ItemConversionError::MissingField("name"))?,
                    input: input.clone(),
                    arguments: arguments.clone(),
                })
            }
            ResponseOutputItem::WebSearchCall {
                id,
                status,
                kind,
                action,
                page,
                signature,
                error,
                ..
            } => Ok(ResponseInputItem::WebSearchCall {
                id: id.clone(),
                status: status.clone(),
                kind: kind.clone(),
                action: action.clone(),
                page: page.clone(),
                signature: signature.clone(),
                error: error.clone(),
            }),
            ResponseOutputItem::ToolSearchCall { .. } => {
                Err(ItemConversionError::Unsupported("tool_search_call"))
            }
            ResponseOutputItem::Unknown => Err(ItemConversionError::Unsupported("unknown item")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn completed_response_body() -> &'static str {
        r#"{
          "id": "387b53c1-7ad2-4259-9298-61b139f5bd20",
          "object": "response",
          "created_at": 1793084842,
          "status": "completed",
          "background": false,
          "model": "deepseek-flash",
          "output": [
            {
              "type": "message",
              "id": "4e706f35-6ea2-468a-bd19-a9c1adc54726",
              "status": "completed",
              "content": [
                {
                  "type": "output_text",
                  "annotations": [],
                  "logprobs": [],
                  "text": "Hello! How can I help you today?"
                }
              ],
              "phase": "final_answer",
              "role": "assistant"
            }
          ],
          "usage": {
            "input_tokens": 7,
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": 11,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 18
          },
          "metadata": {}
        }"#
    }

    fn failed_web_search_body() -> &'static str {
        r#"{
          "id": "resp_ws",
          "object": "response",
          "created_at": 1,
          "status": "completed",
          "model": "deepseek-flash",
          "output": [
            {
              "type": "web_search_call",
              "id": "ws_1",
              "status": "failed",
              "kind": "read_link",
              "action": { "type": "open_page", "url": "https://example.com" },
              "error": { "code": "url_not_accessible" }
            },
            {
              "type": "message",
              "id": "msg_1",
              "status": "completed",
              "role": "assistant",
              "phase": "final_answer",
              "content": [
                {
                  "type": "output_text",
                  "text": "The page was unavailable.",
                  "annotations": [],
                  "logprobs": []
                }
              ]
            }
          ]
        }"#
    }

    #[test]
    fn deserializes_completed_response_example() {
        let response: Response = serde_json::from_str(completed_response_body()).unwrap();
        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(response.output_text(), "Hello! How can I help you today?");
        assert_eq!(response.usage.as_ref().unwrap().total_tokens, 18);
        response.ensure_success().unwrap();
    }

    #[test]
    fn deserializes_failed_web_search_call() {
        let response: Response = serde_json::from_str(failed_web_search_body()).unwrap();

        match &response.output[0] {
            ResponseOutputItem::WebSearchCall {
                status,
                kind,
                error,
                ..
            } => {
                assert_eq!(status.as_ref(), Some(&OutputItemStatus::Failed));
                assert_eq!(kind.as_deref(), Some("read_link"));
                assert_eq!(error.as_ref().unwrap()["code"], "url_not_accessible");
            }
            other => panic!("unexpected item: {other:?}"),
        }

        // A failed tool call must not take the rest of the response down with it.
        assert_eq!(response.output_text(), "The page was unavailable.");
    }

    #[test]
    fn output_text_skips_non_final_phases() {
        let body = json!({
            "id": "resp_phase",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "deepseek-flash",
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{ "type": "output_text", "text": "draft " }]
                },
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{ "type": "output_text", "text": "answer" }]
                },
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": " legacy" }]
                }
            ]
        });
        let response: Response = serde_json::from_value(body).unwrap();
        assert_eq!(response.output_text(), "answer legacy");
    }

    #[test]
    fn ensure_success_rejects_failed_and_incomplete() {
        let body = json!({
            "id": "resp_failed",
            "object": "response",
            "created_at": 1,
            "status": "failed",
            "model": "deepseek-flash",
            "output": [],
            "error": { "code": "server_error", "message": "boom" }
        });
        let response: Response = serde_json::from_value(body).unwrap();
        let failure = response.ensure_success().unwrap_err();
        assert_eq!(failure.status, ResponseStatus::Failed);
        let rendered = failure.to_string();
        assert!(rendered.contains("server_error"));
        assert!(rendered.contains("boom"));

        let body = json!({
            "id": "resp_incomplete",
            "object": "response",
            "created_at": 1,
            "status": "incomplete",
            "model": "deepseek-flash",
            "output": [],
            "incomplete_details": { "reason": "max_output_tokens" }
        });
        let response: Response = serde_json::from_value(body).unwrap();
        let failure = response.ensure_success().unwrap_err();
        assert!(failure.to_string().contains("max_output_tokens"));
    }

    #[test]
    fn unknown_status_values_round_trip() {
        let raw = json!({
            "id": "resp_unknown",
            "object": "response",
            "created_at": 1,
            "status": "cancelled",
            "model": "deepseek-flash",
            "output": []
        });
        let response: Response = serde_json::from_value(raw).unwrap();
        assert_eq!(
            response.status,
            ResponseStatus::Unknown("cancelled".to_owned())
        );
        assert_eq!(
            serde_json::to_value(&response.status).unwrap(),
            json!("cancelled")
        );

        let status: OutputItemStatus = serde_json::from_value(json!("failed")).unwrap();
        assert_eq!(status, OutputItemStatus::Failed);
        let status: OutputItemStatus = serde_json::from_value(json!("new_state")).unwrap();
        assert_eq!(status, OutputItemStatus::Unknown("new_state".to_owned()));
        assert_eq!(serde_json::to_value(&status).unwrap(), json!("new_state"));
    }

    #[test]
    fn to_input_items_replays_replayable_items() {
        let body = json!({
            "id": "resp_replay",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "deepseek-flash",
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{ "type": "output_text", "text": "Sure." }]
                },
                {
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{\"q\":\"x\"}"
                },
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "kind": "read_link",
                    "action": { "type": "open_page", "url": "https://example.com" },
                    "signature": "sig"
                },
                {
                    "type": "tool_search_call",
                    "status": "completed",
                    "call_id": "call_2"
                }
            ]
        });
        let response: Response = serde_json::from_value(body).unwrap();
        let items = response.to_input_items();

        assert_eq!(items.len(), 3);
        match &items[0] {
            ResponseInputItem::Message { role, content } => {
                assert_eq!(*role, MessageRole::Assistant);
                assert!(matches!(content, InputContent::Parts(parts) if parts.len() == 1));
            }
            other => panic!("unexpected item: {other:?}"),
        }
        match &items[1] {
            ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "lookup");
                assert_eq!(arguments, "{\"q\":\"x\"}");
            }
            other => panic!("unexpected item: {other:?}"),
        }
        match &items[2] {
            ResponseInputItem::WebSearchCall {
                kind, signature, ..
            } => {
                assert_eq!(kind.as_deref(), Some("read_link"));
                assert_eq!(signature.as_deref(), Some("sig"));
            }
            other => panic!("unexpected item: {other:?}"),
        }

        match ResponseInputItem::try_from(&response.output[3]) {
            Err(ItemConversionError::Unsupported("tool_search_call")) => {}
            other => panic!("unexpected conversion: {other:?}"),
        }
    }

    #[test]
    fn function_calls_and_annotations_are_exposed() {
        let body = json!({
            "id": "resp_calls",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "deepseek-flash",
            "output": [
                {
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}"
                },
                {
                    "type": "function_call",
                    "status": "completed",
                    "name": "broken"
                },
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "Cited.",
                        "annotations": [{
                            "type": "url_citation",
                            "url": "https://example.com",
                            "start_index": 0,
                            "end_index": 5
                        }]
                    }]
                }
            ]
        });
        let response: Response = serde_json::from_value(body).unwrap();

        let calls = response.function_calls();
        assert_eq!(calls, vec![("call_1", "lookup", "{}")]);

        let annotations = response.annotations();
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0]["url"], "https://example.com");
    }

    #[test]
    fn keeps_reasoning_summary_and_extra_fields() {
        let body = json!({
            "id": "resp_reasoning",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "deepseek-flash",
            "output": [],
            "reasoning": {
                "effort": "high",
                "summary": { "type": "summary_text", "text": "thought" },
                "future_field": 1
            }
        });
        let response: Response = serde_json::from_value(body).unwrap();
        let reasoning = response.reasoning.as_ref().unwrap();
        assert_eq!(
            reasoning.effort,
            Some(super::super::request::ReasoningEffort::High)
        );
        assert_eq!(reasoning.summary.as_ref().unwrap()["text"], "thought");
        assert_eq!(reasoning.extra["future_field"], 1);
    }
}
