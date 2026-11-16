//! Request body for `POST /responses`, including tools and input items.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateResponseRequest {
    pub model: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Input>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponseTool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_check: Option<SafetyCheck>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<TitleConfig>,
}

impl CreateResponseRequest {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: None,
            instructions: None,
            reasoning: None,
            max_output_tokens: None,
            stream: None,
            temperature: None,
            top_p: None,
            text: None,
            tools: None,
            tool_choice: None,
            top_logprobs: None,
            user: None,
            safety_check: None,
            title: None,
        }
    }

    pub fn input_text(mut self, text: impl Into<String>) -> Self {
        self.input = Some(Input::Text(text.into()));
        self
    }

    pub fn input_items(mut self, items: Vec<ResponseInputItem>) -> Self {
        self.input = Some(Input::Items(items));
        self
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning = Some(ReasoningConfig {
            effort: Some(effort),
        });
        self
    }

    pub fn max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = Some(n);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn tools(mut self, tools: Vec<ResponseTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// Model input: plain string (treated as one user message) or item list.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Input {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Low,
    High,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFormat {
    #[serde(rename = "type")]
    pub format_type: TextFormatType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextFormatType {
    Text,
    JsonObject,
    JsonSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCheck {
    pub level: SafetyLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SafetyLevel {
    Low,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitleConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTool {
    #[serde(rename = "type")]
    pub tool_type: ToolType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema object describing function parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

impl ResponseTool {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Option<Value>,
    ) -> Self {
        Self {
            tool_type: ToolType::Function,
            name: Some(name.into()),
            description: Some(description.into()),
            parameters,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
    Custom,
    WebSearch,
    DeepseekWebSearch,
    DeepseekReadUrl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Named(NamedToolChoice),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub choice_type: NamedToolChoiceType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NamedToolChoiceType {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseInputItem {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub item_type: Option<InputItemType>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<InputContent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<ToolOutput>,
}

impl ResponseInputItem {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            item_type: Some(InputItemType::Message),
            role: Some(MessageRole::User),
            content: Some(InputContent::Text(text.into())),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        }
    }

    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            item_type: Some(InputItemType::Message),
            role: Some(MessageRole::System),
            content: Some(InputContent::Text(text.into())),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self {
            item_type: Some(InputItemType::Message),
            role: Some(MessageRole::Assistant),
            content: Some(InputContent::Text(text.into())),
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        }
    }

    pub fn function_call(
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            item_type: Some(InputItemType::FunctionCall),
            role: None,
            content: None,
            call_id: Some(call_id.into()),
            name: Some(name.into()),
            arguments: Some(arguments.into()),
            output: None,
        }
    }

    pub fn function_call_output(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            item_type: Some(InputItemType::FunctionCallOutput),
            role: None,
            content: None,
            call_id: Some(call_id.into()),
            name: None,
            arguments: None,
            output: Some(ToolOutput::Text(output.into())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputItemType {
    Message,
    FunctionCall,
    FunctionCallOutput,
    CustomToolCall,
    CustomToolCallOutput,
    Reasoning,
    WebSearchCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Developer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    Text(String),
    Parts(Vec<ResponseInputContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    Text(String),
    Parts(Vec<ResponseInputContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputContentPart {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
    },
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
    },
    InputFile {
        file_id: String,
    },
    ReasoningText {
        text: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    Original,
    Auto,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_minimal_create_request() {
        let req = CreateResponseRequest::new("deepseek-flash")
            .input_text("Say hello.")
            .max_output_tokens(32)
            .reasoning_effort(ReasoningEffort::None)
            .stream(false);

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "deepseek-flash");
        assert_eq!(json["input"], "Say hello.");
        assert_eq!(json["max_output_tokens"], 32);
        assert_eq!(json["reasoning"]["effort"], "none");
        assert_eq!(json["stream"], false);
    }

    #[test]
    fn serializes_function_tool() {
        let tool = ResponseTool::function(
            "get_weather",
            "Get weather by city",
            Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            })),
        );

        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "get_weather");
        assert!(json.get("function").is_none());
    }
}
