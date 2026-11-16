//! Request body for `POST /responses`, including tools and input items.
//!
//! The types in this module model the DeepSeek Responses API request format.
//! Invalid combinations are either impossible to build (one enum case per input
//! item) or rejected locally by [`CreateResponseRequest::validate`] before a
//! request is sent.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::response::OutputItemStatus;

/// Maximum length of an `input_image.image_url`, per the API docs.
const MAX_IMAGE_URL_CHARS: usize = 8192;
/// Maximum length of a function tool name, per the API docs.
const MAX_TOOL_NAME_LEN: usize = 128;

/// Body for `POST /responses`.
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

    /// Always sent as `false` (or omitted) by the non-streaming client. The
    /// client rejects `Some(true)` with [`super::Error::StreamingNotImplemented`].
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
        self.reasoning = Some(ReasoningConfig::new(effort));
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

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn top_logprobs(mut self, top_logprobs: u32) -> Self {
        self.top_logprobs = Some(top_logprobs);
        self
    }

    pub fn text(mut self, text: TextConfig) -> Self {
        self.text = Some(text);
        self
    }

    pub fn text_format(mut self, format: TextFormat) -> Self {
        self.text = Some(TextConfig::new(format));
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

    pub fn safety_check(mut self, level: SafetyLevel) -> Self {
        self.safety_check = Some(SafetyCheck::new(level));
        self
    }

    pub fn title(mut self, enabled: bool) -> Self {
        self.title = Some(TitleConfig { enabled });
        self
    }

    /// Check the constraints the API enforces with a `400`.
    ///
    /// [`super::DeepSeekClient::create`] calls this before sending, so callers
    /// normally do not need to invoke it directly.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_input()?;
        self.validate_text()?;
        self.validate_reasoning()?;
        self.validate_tools()?;
        self.validate_tool_choice()?;
        Ok(())
    }

    fn validate_input(&self) -> Result<(), String> {
        match &self.input {
            None => {
                if self.instructions.is_none() {
                    return Err("at least one of `input` and `instructions` is required".to_owned());
                }
            }
            Some(Input::Text(_)) => {}
            Some(Input::Items(items)) => {
                if items.is_empty() {
                    return Err("`input` must not be an empty item list".to_owned());
                }
                validate_items(items)?;
            }
        }
        Ok(())
    }

    fn validate_text(&self) -> Result<(), String> {
        if let Some(TextConfig {
            format: Some(TextFormat::JsonSchema { name, schema }),
        }) = &self.text
        {
            if name.trim().is_empty() {
                return Err(
                    "`text.format` with `type: \"json_schema\"` requires a non-empty `name`"
                        .to_owned(),
                );
            }
            if schema.is_null() {
                return Err(
                    "`text.format` with `type: \"json_schema\"` requires a `schema` object"
                        .to_owned(),
                );
            }
        }
        Ok(())
    }

    fn validate_reasoning(&self) -> Result<(), String> {
        if let Some(ReasoningConfig {
            effort: Some(ReasoningEffort::Unknown(value)),
        }) = &self.reasoning
        {
            return Err(format!("unsupported reasoning effort `{value}`"));
        }
        Ok(())
    }

    fn validate_tools(&self) -> Result<(), String> {
        let Some(tools) = &self.tools else {
            return Ok(());
        };

        let mut names = HashSet::new();
        let mut web_search = false;
        let mut deepseek_web_search = false;
        let mut read_url = false;
        let mut custom = false;

        for tool in tools {
            match tool.tool_type {
                ToolType::Function => {
                    let name = tool
                        .name
                        .as_deref()
                        .ok_or_else(|| "a `function` tool requires a `name`".to_owned())?;
                    if !is_valid_function_name(name) {
                        return Err(format!(
                            "invalid function tool name `{name}`: must be non-empty, at most \
                             {MAX_TOOL_NAME_LEN} characters, and match ^[a-zA-Z0-9_-]+$"
                        ));
                    }
                    if !names.insert(name) {
                        return Err(format!("duplicate tool name `{name}`"));
                    }
                }
                ToolType::Custom => {
                    custom = true;
                    let name = tool
                        .name
                        .as_deref()
                        .ok_or_else(|| "a `custom` tool requires a `name`".to_owned())?;
                    if name != "apply_patch" {
                        return Err(format!(
                            "`custom` tool `{name}` is not supported; only `apply_patch` is"
                        ));
                    }
                    if !names.insert(name) {
                        return Err(format!("duplicate tool name `{name}`"));
                    }
                }
                ToolType::WebSearch => web_search = true,
                ToolType::DeepseekWebSearch => deepseek_web_search = true,
                ToolType::DeepseekReadUrl => read_url = true,
            }
        }

        if web_search && (deepseek_web_search || read_url) {
            return Err(
                "`web_search` cannot be declared together with `deepseek_web_search` or \
                 `deepseek_read_url`"
                    .to_owned(),
            );
        }

        let uses_web_tools = web_search || deepseek_web_search || read_url;
        if uses_web_tools && custom {
            return Err("`custom` tools cannot be declared together with the web tools".to_owned());
        }

        if uses_web_tools {
            if self.top_logprobs.is_some() {
                return Err(
                    "`top_logprobs` is not supported together with the web tools".to_owned(),
                );
            }
            if let Some(TextConfig {
                format: Some(text_format),
            }) = &self.text
                && !matches!(text_format, TextFormat::Text)
            {
                return Err(
                    "`text.format` other than `text` is not supported together with the web tools"
                        .to_owned(),
                );
            }
            if let Some(Input::Items(items)) = &self.input
                && matches!(
                    items.last(),
                    Some(ResponseInputItem::Message {
                        role: MessageRole::Assistant,
                        ..
                    }) | Some(ResponseInputItem::Reasoning { .. })
                )
            {
                return Err(
                    "`input` must not end with an assistant message or a reasoning item when the \
                     web tools are declared"
                        .to_owned(),
                );
            }
        }

        Ok(())
    }

    fn validate_tool_choice(&self) -> Result<(), String> {
        if let Some(ToolChoice::Named(choice)) = &self.tool_choice
            && choice.name.trim().is_empty()
        {
            return Err("a named `tool_choice` requires a non-empty `name`".to_owned());
        }
        Ok(())
    }
}

/// Model input: a plain string (treated as one `user` message) or an item list.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Input {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

/// Configuration of the thinking mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
}

impl ReasoningConfig {
    pub fn new(effort: ReasoningEffort) -> Self {
        Self {
            effort: Some(effort),
        }
    }
}

/// Thinking effort.
///
/// `minimal`, `medium` and `xhigh` are accepted by the API for compatibility
/// with existing clients (they are mapped to `low` or `high`). Truly unknown
/// values are kept for decoding tolerance and rejected when a request is sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Unknown(String),
}

impl ReasoningEffort {
    pub fn as_str(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
            Self::Unknown(value) => value,
        }
    }
}

impl From<String> for ReasoningEffort {
    fn from(value: String) -> Self {
        match value.as_str() {
            "none" => Self::None,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            "max" => Self::Max,
            _ => Self::Unknown(value),
        }
    }
}

impl From<ReasoningEffort> for String {
    fn from(value: ReasoningEffort) -> Self {
        value.as_str().to_owned()
    }
}

/// Text output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

impl TextConfig {
    pub fn new(format: TextFormat) -> Self {
        Self {
            format: Some(format),
        }
    }
}

/// Output format. `JsonSchema` carries the required `name` and `schema`, so an
/// invalid `json_schema` body is impossible to build.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    Text,
    JsonObject,
    JsonSchema { name: String, schema: Value },
}

impl TextFormat {
    pub fn json_schema(name: impl Into<String>, schema: Value) -> Self {
        Self::JsonSchema {
            name: name.into(),
            schema,
        }
    }
}

/// Model content safety (DeepSeek extension).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCheck {
    pub level: SafetyLevel,
}

impl SafetyCheck {
    pub fn new(level: SafetyLevel) -> Self {
        Self { level }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SafetyLevel {
    Low,
    High,
}

/// Ask the response to carry a short conversation title (DeepSeek extension).
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

    /// The `apply_patch` custom tool (Codex compatibility).
    pub fn custom(name: impl Into<String>) -> Self {
        Self {
            tool_type: ToolType::Custom,
            name: Some(name.into()),
            description: None,
            parameters: None,
        }
    }

    pub fn web_search() -> Self {
        Self {
            tool_type: ToolType::WebSearch,
            name: None,
            description: None,
            parameters: None,
        }
    }

    pub fn deepseek_web_search() -> Self {
        Self {
            tool_type: ToolType::DeepseekWebSearch,
            name: None,
            description: None,
            parameters: None,
        }
    }

    pub fn deepseek_read_url() -> Self {
        Self {
            tool_type: ToolType::DeepseekReadUrl,
            name: None,
            description: None,
            parameters: None,
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

/// Forces the model to call a specific function. `name` is required by the API
/// whenever `type` is `function`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub choice_type: NamedToolChoiceType,
    pub name: String,
}

impl NamedToolChoice {
    pub fn function(name: impl Into<String>) -> Self {
        Self {
            choice_type: NamedToolChoiceType::Function,
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NamedToolChoiceType {
    Function,
}

/// One item of the `input` list.
///
/// Each item type is its own variant, so a `function_call` can never be built
/// without a `call_id` and a `message` can never be built without a role.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    /// A chat message. `content` may be a plain string or content parts.
    Message {
        role: MessageRole,
        content: InputContent,
    },

    /// A previous function call. Must be paired with a `FunctionCallOutput`.
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },

    /// The result of a previous function call.
    FunctionCallOutput { call_id: String, output: ToolOutput },

    /// A previous `apply_patch` custom tool call.
    CustomToolCall {
        call_id: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        input: Option<String>,
        /// Some servers send the payload as `arguments`; kept for round-trips.
        #[serde(skip_serializing_if = "Option::is_none")]
        arguments: Option<String>,
    },

    /// The result of a previous custom tool call.
    CustomToolCallOutput { call_id: String, output: ToolOutput },

    /// Chain-of-thought to replay. Only `reasoning_text` parts are accepted.
    Reasoning {
        content: Vec<ResponseInputContentPart>,
    },

    /// A `web_search_call` returned by a previous response.
    ///
    /// Pass the item back unchanged (including `kind` and `signature`) so its
    /// search results or read pages are restored into the context.
    WebSearchCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<OutputItemStatus>,
        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        action: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        page: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<Value>,
    },
}

impl ResponseInputItem {
    pub fn message(role: MessageRole, content: impl Into<InputContent>) -> Self {
        Self::Message {
            role,
            content: content.into(),
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::message(MessageRole::User, text.into())
    }

    pub fn system_text(text: impl Into<String>) -> Self {
        Self::message(MessageRole::System, text.into())
    }

    pub fn developer_text(text: impl Into<String>) -> Self {
        Self::message(MessageRole::Developer, text.into())
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::message(MessageRole::Assistant, text.into())
    }

    pub fn function_call(
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::FunctionCall {
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    pub fn function_call_output(call_id: impl Into<String>, output: impl Into<ToolOutput>) -> Self {
        Self::FunctionCallOutput {
            call_id: call_id.into(),
            output: output.into(),
        }
    }

    pub fn custom_tool_call(
        call_id: impl Into<String>,
        name: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
        Self::CustomToolCall {
            call_id: call_id.into(),
            name: name.into(),
            input: Some(input.into()),
            arguments: None,
        }
    }

    pub fn custom_tool_call_output(
        call_id: impl Into<String>,
        output: impl Into<ToolOutput>,
    ) -> Self {
        Self::CustomToolCallOutput {
            call_id: call_id.into(),
            output: output.into(),
        }
    }

    pub fn reasoning_text(text: impl Into<String>) -> Self {
        Self::reasoning(vec![ResponseInputContentPart::reasoning_text(text)])
    }

    pub fn reasoning(parts: Vec<ResponseInputContentPart>) -> Self {
        Self::Reasoning { content: parts }
    }
}

fn validate_items(items: &[ResponseInputItem]) -> Result<(), String> {
    let mut call_ids: HashSet<&str> = HashSet::new();
    let mut output_ids: HashSet<&str> = HashSet::new();

    for (index, item) in items.iter().enumerate() {
        match item {
            ResponseInputItem::Message { role, content } => {
                validate_message(role, content)
                    .map_err(|message| format!("input[{index}]: {message}"))?;
            }
            ResponseInputItem::Reasoning { content } => {
                if content.is_empty() {
                    return Err(format!(
                        "input[{index}]: reasoning content must not be empty"
                    ));
                }
                for part in content {
                    if !matches!(part, ResponseInputContentPart::ReasoningText { .. }) {
                        return Err(format!(
                            "input[{index}]: reasoning items only accept `reasoning_text` parts"
                        ));
                    }
                }
            }
            ResponseInputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                validate_call_id(call_id)
                    .map_err(|message| format!("input[{index}]: {message}"))?;
                if name.is_empty() {
                    return Err(format!("input[{index}]: `function_call` requires a `name`"));
                }
                if arguments.is_empty() {
                    return Err(format!(
                        "input[{index}]: `function_call` requires non-empty `arguments`"
                    ));
                }
                if !call_ids.insert(call_id.as_str()) {
                    return Err(format!("input[{index}]: duplicate `call_id` `{call_id}`"));
                }
            }
            ResponseInputItem::FunctionCallOutput { call_id, output } => {
                validate_call_id(call_id)
                    .map_err(|message| format!("input[{index}]: {message}"))?;
                validate_tool_output(output)
                    .map_err(|message| format!("input[{index}]: {message}"))?;
                if !output_ids.insert(call_id.as_str()) {
                    return Err(format!(
                        "input[{index}]: duplicate `function_call_output` for `call_id` `{call_id}`"
                    ));
                }
            }
            ResponseInputItem::CustomToolCall {
                call_id,
                name,
                input,
                arguments,
            } => {
                validate_call_id(call_id)
                    .map_err(|message| format!("input[{index}]: {message}"))?;
                if name.is_empty() {
                    return Err(format!(
                        "input[{index}]: `custom_tool_call` requires a `name`"
                    ));
                }
                if input.is_none() && arguments.is_none() {
                    return Err(format!(
                        "input[{index}]: `custom_tool_call` requires `input` or `arguments`"
                    ));
                }
                if !call_ids.insert(call_id.as_str()) {
                    return Err(format!("input[{index}]: duplicate `call_id` `{call_id}`"));
                }
            }
            ResponseInputItem::CustomToolCallOutput { call_id, output } => {
                validate_call_id(call_id)
                    .map_err(|message| format!("input[{index}]: {message}"))?;
                validate_tool_output(output)
                    .map_err(|message| format!("input[{index}]: {message}"))?;
                if !output_ids.insert(call_id.as_str()) {
                    return Err(format!(
                        "input[{index}]: duplicate `custom_tool_call_output` for `call_id` \
                         `{call_id}`"
                    ));
                }
            }
            ResponseInputItem::WebSearchCall { .. } => {}
        }
    }

    for call_id in call_ids {
        if !output_ids.contains(call_id) {
            return Err(format!(
                "tool call `{call_id}` has no matching call output item"
            ));
        }
    }

    Ok(())
}

fn validate_message(role: &MessageRole, content: &InputContent) -> Result<(), String> {
    let InputContent::Parts(parts) = content else {
        return Ok(());
    };

    for part in parts {
        match part {
            ResponseInputContentPart::InputImage {
                image_url, file_id, ..
            } => {
                if matches!(role, MessageRole::Assistant) {
                    return Err(
                        "`input_image` parts are not allowed in assistant messages".to_owned()
                    );
                }
                validate_image_source(image_url.as_deref(), file_id.as_deref())?;
            }
            ResponseInputContentPart::InputFile { file_id } => {
                if !matches!(role, MessageRole::User) {
                    return Err("`input_file` parts are only allowed in user messages".to_owned());
                }
                if file_id.is_empty() {
                    return Err("`input_file` requires a non-empty `file_id`".to_owned());
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn validate_tool_output(output: &ToolOutput) -> Result<(), String> {
    let ToolOutput::Parts(parts) = output else {
        return Ok(());
    };

    for part in parts {
        match part {
            ResponseInputContentPart::InputImage {
                image_url, file_id, ..
            } => validate_image_source(image_url.as_deref(), file_id.as_deref())?,
            ResponseInputContentPart::InputFile { .. } => {
                return Err("`input_file` parts are not allowed in tool output".to_owned());
            }
            _ => {}
        }
    }

    Ok(())
}

fn validate_image_source(image_url: Option<&str>, file_id: Option<&str>) -> Result<(), String> {
    match (image_url, file_id) {
        (None, None) => Err("`input_image` must have either `image_url` or `file_id`".to_owned()),
        (Some(_), Some(_)) => {
            Err("`input_image` cannot have both `image_url` and `file_id`".to_owned())
        }
        (Some(url), None) => {
            if url.is_empty() {
                return Err("`image_url` must not be empty".to_owned());
            }
            if url.chars().count() > MAX_IMAGE_URL_CHARS {
                return Err(format!(
                    "`image_url` exceeds {MAX_IMAGE_URL_CHARS} characters"
                ));
            }
            Ok(())
        }
        (None, Some(file_id)) => {
            if file_id.is_empty() {
                return Err("`file_id` must not be empty".to_owned());
            }
            Ok(())
        }
    }
}

fn validate_call_id(call_id: &str) -> Result<(), String> {
    if call_id.is_empty() {
        Err("`call_id` must not be empty".to_owned())
    } else {
        Ok(())
    }
}

fn is_valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_LEN
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// A chat message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Developer,
}

/// Message content: a plain string or content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    Text(String),
    Parts(Vec<ResponseInputContentPart>),
}

impl From<String> for InputContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for InputContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<Vec<ResponseInputContentPart>> for InputContent {
    fn from(value: Vec<ResponseInputContentPart>) -> Self {
        Self::Parts(value)
    }
}

/// Tool output: a plain string or content parts (text and images).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    Text(String),
    Parts(Vec<ResponseInputContentPart>),
}

impl From<String> for ToolOutput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ToolOutput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<Vec<ResponseInputContentPart>> for ToolOutput {
    fn from(value: Vec<ResponseInputContentPart>) -> Self {
        Self::Parts(value)
    }
}

/// One content part of a message or tool output.
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

impl ResponseInputContentPart {
    pub fn input_text(text: impl Into<String>) -> Self {
        Self::InputText { text: text.into() }
    }

    pub fn output_text(text: impl Into<String>) -> Self {
        Self::OutputText { text: text.into() }
    }

    pub fn image_url(image_url: impl Into<String>) -> Self {
        Self::InputImage {
            image_url: Some(image_url.into()),
            detail: None,
            file_id: None,
        }
    }

    pub fn image_file(file_id: impl Into<String>) -> Self {
        Self::InputImage {
            image_url: None,
            detail: None,
            file_id: Some(file_id.into()),
        }
    }

    pub fn input_file(file_id: impl Into<String>) -> Self {
        Self::InputFile {
            file_id: file_id.into(),
        }
    }

    pub fn reasoning_text(text: impl Into<String>) -> Self {
        Self::ReasoningText { text: text.into() }
    }
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
    use serde_json::json;

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
        req.validate().unwrap();
    }

    #[test]
    fn serializes_function_tool() {
        let tool = ResponseTool::function(
            "get_weather",
            "Get weather by city",
            Some(json!({
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

    #[test]
    fn serializes_function_call_history() {
        let items = vec![
            ResponseInputItem::user_text("What is the weather?"),
            ResponseInputItem::function_call("call_1", "get_weather", r#"{"city":"Hangzhou"}"#),
            ResponseInputItem::function_call_output("call_1", "24 C"),
        ];
        let request = CreateResponseRequest::new("deepseek-flash").input_items(items);
        request.validate().unwrap();

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["input"][0]["type"], "message");
        assert_eq!(json["input"][0]["role"], "user");
        assert_eq!(json["input"][1]["type"], "function_call");
        assert_eq!(json["input"][1]["call_id"], "call_1");
        assert_eq!(json["input"][1]["arguments"], r#"{"city":"Hangzhou"}"#);
        assert_eq!(json["input"][2]["type"], "function_call_output");
        assert_eq!(json["input"][2]["output"], "24 C");
    }

    #[test]
    fn serializes_web_search_call_replay() {
        let item = ResponseInputItem::WebSearchCall {
            id: Some("ws_1".to_owned()),
            status: Some(OutputItemStatus::Completed),
            kind: Some("read_link".to_owned()),
            action: Some(json!({ "type": "open_page", "url": "https://example.com" })),
            page: Some(json!({ "url": "https://example.com", "title": "Example" })),
            signature: Some("encrypted-page".to_owned()),
            error: None,
        };
        let request = CreateResponseRequest::new("deepseek-flash").input_items(vec![
            ResponseInputItem::user_text("Summarize https://example.com"),
            item,
        ]);
        request.validate().unwrap();

        let json = serde_json::to_value(&request).unwrap();
        let replay = &json["input"][1];
        assert_eq!(replay["type"], "web_search_call");
        assert_eq!(replay["id"], "ws_1");
        assert_eq!(replay["status"], "completed");
        assert_eq!(replay["kind"], "read_link");
        assert_eq!(replay["signature"], "encrypted-page");
        assert_eq!(replay["action"]["url"], "https://example.com");
        assert!(replay.get("error").is_none());
    }

    #[test]
    fn serializes_custom_tool_call() {
        let item = ResponseInputItem::custom_tool_call(
            "call_patch",
            "apply_patch",
            "*** Begin Patch\n*** End Patch",
        );
        let output = ResponseInputItem::custom_tool_call_output("call_patch", "patch applied");
        let request = CreateResponseRequest::new("deepseek-flash").input_items(vec![item, output]);
        request.validate().unwrap();

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["input"][0]["type"], "custom_tool_call");
        assert_eq!(json["input"][0]["name"], "apply_patch");
        assert_eq!(json["input"][0]["input"], "*** Begin Patch\n*** End Patch");
        assert!(json["input"][0].get("arguments").is_none());
        assert_eq!(json["input"][1]["type"], "custom_tool_call_output");
        assert_eq!(json["input"][1]["call_id"], "call_patch");
    }

    #[test]
    fn rejects_missing_input_and_instructions() {
        let err = CreateResponseRequest::new("deepseek-flash")
            .validate()
            .unwrap_err();
        assert!(err.contains("at least one"));
    }

    #[test]
    fn rejects_empty_item_list() {
        let err = CreateResponseRequest::new("deepseek-flash")
            .input_items(Vec::new())
            .validate()
            .unwrap_err();
        assert!(err.contains("empty item list"));
    }

    #[test]
    fn rejects_blank_json_schema_name() {
        let request = CreateResponseRequest::new("deepseek-flash")
            .input_text("hello")
            .text_format(TextFormat::json_schema("  ", json!({ "type": "object" })));
        let err = request.validate().unwrap_err();
        assert!(err.contains("non-empty `name`"));
    }

    #[test]
    fn rejects_input_image_without_source() {
        let item = ResponseInputItem::message(
            MessageRole::User,
            InputContent::Parts(vec![ResponseInputContentPart::InputImage {
                image_url: None,
                detail: None,
                file_id: None,
            }]),
        );
        let err = CreateResponseRequest::new("deepseek-flash")
            .input_items(vec![item])
            .validate()
            .unwrap_err();
        assert!(err.contains("either `image_url` or `file_id`"));
    }

    #[test]
    fn rejects_input_image_with_both_sources() {
        let item = ResponseInputItem::message(
            MessageRole::User,
            InputContent::Parts(vec![ResponseInputContentPart::InputImage {
                image_url: Some("https://example.com/a.png".to_owned()),
                detail: None,
                file_id: Some("file-api-1".to_owned()),
            }]),
        );
        let err = CreateResponseRequest::new("deepseek-flash")
            .input_items(vec![item])
            .validate()
            .unwrap_err();
        assert!(err.contains("both `image_url` and `file_id`"));
    }

    #[test]
    fn rejects_input_image_in_assistant_message() {
        let item = ResponseInputItem::message(
            MessageRole::Assistant,
            InputContent::Parts(vec![ResponseInputContentPart::image_url(
                "https://example.com/a.png",
            )]),
        );
        let err = CreateResponseRequest::new("deepseek-flash")
            .input_items(vec![item])
            .validate()
            .unwrap_err();
        assert!(err.contains("assistant"));
    }

    #[test]
    fn rejects_unpaired_function_call() {
        let items = vec![ResponseInputItem::function_call(
            "call_1",
            "get_weather",
            "{}",
        )];
        let err = CreateResponseRequest::new("deepseek-flash")
            .input_items(items)
            .validate()
            .unwrap_err();
        assert!(err.contains("no matching call output"));
    }

    #[test]
    fn rejects_web_search_tool_conflicts() {
        let err = CreateResponseRequest::new("deepseek-flash")
            .input_text("hi")
            .tools(vec![
                ResponseTool::web_search(),
                ResponseTool::deepseek_web_search(),
            ])
            .validate()
            .unwrap_err();
        assert!(err.contains("cannot be declared together"));

        let err = CreateResponseRequest::new("deepseek-flash")
            .input_text("hi")
            .tools(vec![ResponseTool::deepseek_web_search()])
            .top_logprobs(5)
            .validate()
            .unwrap_err();
        assert!(err.contains("top_logprobs"));
    }

    #[test]
    fn rejects_unknown_reasoning_effort() {
        let mut request = CreateResponseRequest::new("deepseek-flash").input_text("hello");
        request.reasoning = Some(ReasoningConfig::new(ReasoningEffort::Unknown(
            "turbo".to_owned(),
        )));
        let err = request.validate().unwrap_err();
        assert!(err.contains("unsupported reasoning effort"));
    }

    #[test]
    fn rejects_duplicate_function_names() {
        let err = CreateResponseRequest::new("deepseek-flash")
            .input_text("hi")
            .tools(vec![
                ResponseTool::function("lookup", "first", None),
                ResponseTool::function("lookup", "second", None),
            ])
            .validate()
            .unwrap_err();
        assert!(err.contains("duplicate tool name"));
    }

    #[test]
    fn accepts_complete_request() {
        let request = CreateResponseRequest::new("deepseek-flash")
            .instructions("You are helpful.")
            .input_items(vec![ResponseInputItem::user_text("hello")])
            .reasoning_effort(ReasoningEffort::High)
            .text_format(TextFormat::json_schema(
                "answer",
                json!({ "type": "object", "properties": {} }),
            ))
            .tools(vec![ResponseTool::function(
                "lookup",
                "Look things up",
                None,
            )])
            .tool_choice(ToolChoice::Named(NamedToolChoice::function("lookup")))
            .safety_check(SafetyLevel::Low)
            .title(true);
        request.validate().unwrap();
    }

    #[test]
    fn reasoning_effort_round_trips_compatibility_aliases() {
        for (raw, expected) in [
            ("none", ReasoningEffort::None),
            ("minimal", ReasoningEffort::Minimal),
            ("low", ReasoningEffort::Low),
            ("medium", ReasoningEffort::Medium),
            ("high", ReasoningEffort::High),
            ("xhigh", ReasoningEffort::XHigh),
            ("max", ReasoningEffort::Max),
        ] {
            let parsed: ReasoningEffort = serde_json::from_value(json!(raw)).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(serde_json::to_value(&parsed).unwrap(), json!(raw));
        }
    }
}
