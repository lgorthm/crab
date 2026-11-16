//! Model response from `POST /responses`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::request::{MessageRole, ReasoningConfig, ResponseTool, ToolChoice};

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
    pub reasoning: Option<ReasoningConfig>,

    /// Catch-all for fields we do not model yet (store, metadata, etc.).
    #[serde(default, flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl Response {
    /// Concatenate all `output_text` parts from completed message items.
    pub fn output_text(&self) -> String {
        self.output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::Message { content, .. } => Some(content),
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

    /// All function_call items in this response.
    pub fn function_calls(&self) -> Vec<&ResponseOutputItem> {
        self.output
            .iter()
            .filter(|item| matches!(item, ResponseOutputItem::FunctionCall { .. }))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteReason {
    MaxOutputTokens,
    ContentFilter,
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
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputItemStatus {
    InProgress,
    Completed,
    Incomplete,
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
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,

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
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
    #[serde(default)]
    pub safety_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerToolUse {
    pub tool_search: u32,
    pub tool_fetch: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_image_search: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_completed_response_example() {
        let raw = r#"{
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
        }"#;

        let response: Response = serde_json::from_str(raw).unwrap();
        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(response.output_text(), "Hello! How can I help you today?");
        assert_eq!(response.usage.as_ref().unwrap().total_tokens, 18);
    }
}
