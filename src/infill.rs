use std::{env, iter, sync::Arc};

use anyhow::Result;
use futures_util::StreamExt;
use ramhorns::Content;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};

use crate::config::{CompletionConfig, CopilotConfig, ModelConfig, TemplateConfig};

impl From<CompletionConfig> for InfillProvider {
  fn from(config: CompletionConfig) -> Self {
    match config {
      CompletionConfig::Empty => Self::Empty,
      CompletionConfig::Mistral { config } => Self::Mistral(config),
      CompletionConfig::LlamaCpp { config } => Self::LlamaCpp(config),
      CompletionConfig::Ollama { config } => Self::Ollama(config),
      CompletionConfig::OpenAICompletions { config, template } => Self::OpenAICompletions(template, config),
      CompletionConfig::Copilot { config } => Self::Copilot(config),
    }
  }
}

pub enum InfillProvider {
  Empty,
  Mistral(Arc<ModelConfig>),
  LlamaCpp(Arc<ModelConfig>),
  Ollama(Arc<ModelConfig>),
  OpenAICompletions(Arc<TemplateConfig>, Arc<ModelConfig>),
  Copilot(Arc<CopilotConfig>),
}

impl InfillProvider {
  pub async fn infill(
    &self,
    client: Arc<Client>,
    prefix: String,
    suffix: String,
    language_id: String,
  ) -> Result<Box<dyn Iterator<Item = String> + Send>> {
    Ok(match self {
      Self::Empty => Box::new(iter::empty()),
      Self::Mistral(config) => mistral(client, config, prefix, suffix).await?,
      Self::LlamaCpp(config) => llama_cpp(client, config, prefix, suffix).await?,
      Self::Ollama(config) => ollama(client, config, prefix, suffix).await?,
      Self::OpenAICompletions(template, config) => openai(client, template.clone(), config, prefix, suffix).await?,
      Self::Copilot(config) => copilot(client, config, prefix, suffix, language_id).await?,
    })
  }
}

fn with_optional_auth(request: RequestBuilder, api_key_env: &Option<String>) -> Result<RequestBuilder> {
  Ok(match api_key_env {
    Some(env_name) => request.bearer_auth(env::var(env_name)?),
    None => request,
  })
}

async fn mistral(
  client: Arc<Client>,
  config: &Arc<ModelConfig>,
  prefix: String,
  suffix: String,
) -> Result<Box<dyn Iterator<Item = String> + Send>> {
  let response = with_optional_auth(client.post(&config.url), &config.api_key_env)?
    .json(&MistralRequest {
      model: config.generation_config.model.as_deref().unwrap_or(""),
      prompt: prefix,
      suffix: Some(suffix),
      temperature: config.generation_config.temperature,
      top_p: config.generation_config.top_p,
      max_tokens: config.generation_config.max_tokens,
      min_tokens: config.generation_config.min_tokens,
      stop: &config.generation_config.stop,
      random_seed: config.generation_config.seed,
    })
    .send()
    .await?
    .json::<MistralResponse>()
    .await?;
  Ok(Box::new(response.choices.into_iter().map(|c| c.message.content)))
}

async fn llama_cpp(
  client: Arc<Client>,
  config: &Arc<ModelConfig>,
  prefix: String,
  suffix: String,
) -> Result<Box<dyn Iterator<Item = String> + Send>> {
  let response = with_optional_auth(client.post(&config.url), &config.api_key_env)?
    .json(&LlamaCppRequest {
      input_prefix: prefix,
      input_suffix: suffix,
      temperature: config.generation_config.temperature,
      top_p: config.generation_config.top_p,
      max_tokens: config.generation_config.max_tokens,
      stop: &config.generation_config.stop,
      seed: config.generation_config.seed,
    })
    .send()
    .await?
    .json::<LlamaCppResponse>()
    .await?;
  Ok(Box::new(iter::once(response.content)))
}

async fn ollama(
  client: Arc<Client>,
  config: &Arc<ModelConfig>,
  prefix: String,
  suffix: String,
) -> Result<Box<dyn Iterator<Item = String> + Send>> {
  let response = with_optional_auth(client.post(&config.url), &config.api_key_env)?
    .json(&OllamaRequest {
      model: config.generation_config.model.as_deref().unwrap_or(""),
      prompt: prefix,
      suffix,
      stream: true,
      options: OllamaOptions {
        temperature: config.generation_config.temperature,
        stop: &config.generation_config.stop,
        num_predict: config.generation_config.max_tokens,
        seed: config.generation_config.seed,
      },
    })
    .send()
    .await?;

  let mut stream = response.bytes_stream();
  let mut buf = String::new();
  let mut completion = String::new();
  while let Some(chunk) = stream.next().await {
    buf.push_str(std::str::from_utf8(chunk?.as_ref())?);
    while let Some(newline) = buf.find('\n') {
      let line = buf[..newline].trim_end_matches('\r').to_string();
      buf = buf[newline + 1..].to_string();
      if line.trim().is_empty() {
        continue;
      }
      let chunk: OllamaStreamChunk = serde_json::from_str(&line)?;
      if let Some(text) = chunk.response {
        completion.push_str(&text);
      }
      if chunk.done {
        return Ok(Box::new(iter::once(completion)));
      }
    }
  }
  if !buf.trim().is_empty()
    && let Ok(chunk) = serde_json::from_str::<OllamaStreamChunk>(buf.trim())
    && let Some(text) = chunk.response
  {
    completion.push_str(&text);
  }
  Ok(Box::new(iter::once(completion)))
}

async fn openai(
  client: Arc<Client>,
  template: Arc<TemplateConfig>,
  config: &Arc<ModelConfig>,
  prefix: String,
  suffix: String,
) -> Result<Box<dyn Iterator<Item = String> + Send>> {
  let prompt = template.0.render(&OpenAIContent { prefix, suffix });
  let response = with_optional_auth(client.post(&config.url), &config.api_key_env)?
    .json(&OpenAIRequest {
      model: &config.generation_config.model,
      prompt,
      temperature: config.generation_config.temperature,
      top_p: config.generation_config.top_p,
      max_tokens: config.generation_config.max_tokens,
      stop: &config.generation_config.stop,
      seed: config.generation_config.seed,
    })
    .send()
    .await?
    .json::<OpenAIResponse>()
    .await?;
  Ok(Box::new(response.choices.into_iter().map(|c| c.text)))
}

async fn copilot(
  client: Arc<Client>,
  config: &Arc<CopilotConfig>,
  prefix: String,
  suffix: String,
  language_id: String,
) -> Result<Box<dyn Iterator<Item = String> + Send>> {
  let token = env::var(&config.api_key_env)?;
  let request_id = uuid::Uuid::new_v4().to_string();
  let prompt_tokens = prefix.split_whitespace().count();
  let suffix_tokens = suffix.split_whitespace().count();
  let response = client
    .post("https://copilot-proxy.githubusercontent.com/v1/engines/copilot-codex/completions")
    .header("User-Agent", "GitHubCopilotChat/0.0.1")
    .header("Authorization", format!("Bearer {token}"))
    .header("X-Request-Id", &request_id)
    .header("X-GitHub-Api-Version", "2025-04-01")
    .header("Openai-Organization", "github-copilot")
    .header("VScode-SessionId", &uuid::Uuid::new_v4().to_string())
    .header("VScode-MachineId", &uuid::Uuid::new_v4().to_string())
    .header("OpenAI-Intent", "copilot-ghost")
    .json(&CopilotRequest {
      prompt: &prefix,
      suffix: &suffix,
      max_tokens: config.max_tokens,
      temperature: config.temperature.unwrap_or(0.0),
      top_p: config.top_p.unwrap_or(1.0),
      n: 1,
      stop: &config.stop,
      stream: true,
      extra: CopilotExtra {
        language: &language_id,
        next_indent: 0,
        trim_by_indentation: true,
        prompt_tokens,
        suffix_tokens,
      },
    })
    .send()
    .await?;

  let mut stream = response.bytes_stream();
  let mut buf = String::new();
  let mut completions: Vec<String> = Vec::new();
  while let Some(chunk) = stream.next().await {
    buf.push_str(std::str::from_utf8(chunk?.as_ref())?);
    while let Some(newline) = buf.find('\n') {
      let line = buf[..newline].trim_end_matches('\r').to_string();
      buf = buf[newline + 1..].to_string();
      if let Some(data) = line.strip_prefix("data: ") {
        if data == "[DONE]" {
          break;
        }
        let chunk: CopilotChunk = serde_json::from_str(data)?;
        for choice in chunk.choices {
          match completions.first_mut() {
            Some(s) => s.push_str(&choice.text),
            None => completions.push(choice.text),
          }
        }
      }
    }
  }
  Ok(Box::new(completions.into_iter()))
}

#[derive(Serialize)]
struct MistralRequest<'a> {
  model: &'a str,
  prompt: String,
  suffix: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  top_p: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  min_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: &'a Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  random_seed: Option<u32>,
}

#[derive(Deserialize)]
struct MistralResponse {
  choices: Vec<MistralChoice>,
}
#[derive(Deserialize)]
struct MistralChoice {
  message: MistralMessage,
}
#[derive(Deserialize)]
struct MistralMessage {
  content: String,
}

#[derive(Serialize)]
struct LlamaCppRequest<'a> {
  input_prefix: String,
  input_suffix: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  top_p: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: &'a Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  seed: Option<u32>,
}

#[derive(Deserialize)]
struct LlamaCppResponse {
  content: String,
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
  model: &'a str,
  prompt: String,
  suffix: String,
  stream: bool,
  options: OllamaOptions<'a>,
}

#[derive(Serialize)]
struct OllamaOptions<'a> {
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f64>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: &'a Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  num_predict: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  seed: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaStreamChunk {
  #[serde(default)]
  response: Option<String>,
  #[serde(default)]
  done: bool,
}

#[derive(Content)]
struct OpenAIContent {
  prefix: String,
  suffix: String,
}

#[derive(Serialize)]
struct OpenAIRequest<'a> {
  model: &'a Option<String>,
  prompt: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  top_p: Option<f64>,
  #[serde(skip_serializing_if = "Option::is_none")]
  max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: &'a Vec<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  seed: Option<u32>,
}

#[derive(Deserialize)]
struct OpenAIResponse {
  choices: Vec<OpenAIChoice>,
}
#[derive(Deserialize)]
struct OpenAIChoice {
  text: String,
}

#[derive(Serialize)]
struct CopilotRequest<'a> {
  prompt: &'a str,
  suffix: &'a str,
  #[serde(skip_serializing_if = "Option::is_none")]
  max_tokens: Option<u32>,
  temperature: f64,
  top_p: f64,
  n: u32,
  #[serde(skip_serializing_if = "Vec::is_empty")]
  stop: &'a Vec<String>,
  stream: bool,
  extra: CopilotExtra<'a>,
}

#[derive(Serialize)]
struct CopilotExtra<'a> {
  language: &'a str,
  next_indent: u32,
  trim_by_indentation: bool,
  prompt_tokens: usize,
  suffix_tokens: usize,
}

#[derive(Deserialize)]
struct CopilotChunk {
  choices: Vec<CopilotChoice>,
}
#[derive(Deserialize)]
struct CopilotChoice {
  #[serde(default)]
  text: String,
}
