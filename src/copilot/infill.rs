use std::{
  env,
  sync::Arc,
  time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{config::CopilotConfig, infill::Infill};

const TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const COMPLETIONS_URL: &str = "https://copilot-proxy.githubusercontent.com/v1/engines/copilot-codex/completions";
const USER_AGENT: &str = "GitHubCopilotChat/0.0.1";

#[derive(Debug, Deserialize)]
struct CopilotTokenResponse {
  token: String,
  expires_at: u64,
}

#[derive(Debug)]
struct CachedToken {
  token: String,
  expires_at: u64,
}

#[derive(Debug, Serialize)]
struct CopilotCompletionsExtra<'a> {
  language: &'a str,
  next_indent: u32,
  trim_by_indentation: bool,
  prompt_tokens: usize,
  suffix_tokens: usize,
}

#[derive(Debug, Serialize)]
struct CopilotCompletionsRequest<'a> {
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
  extra: CopilotCompletionsExtra<'a>,
}

#[derive(Debug, Deserialize)]
struct CompletionsChoice {
  #[serde(default)]
  text: String,
}

#[derive(Debug, Deserialize)]
struct CompletionsChunk {
  choices: Vec<CompletionsChoice>,
}

fn now_seconds() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs()
}

#[derive(Debug)]
pub struct CopilotInfill {
  config: Arc<CopilotConfig>,
  cached_token: Mutex<Option<CachedToken>>,
  session_id: String,
  machine_id: String,
}

impl CopilotInfill {
  pub fn new(config: Arc<CopilotConfig>) -> Self {
    Self {
      config,
      cached_token: Mutex::new(None),
      session_id: Uuid::new_v4().to_string(),
      machine_id: Uuid::new_v4().to_string(),
    }
  }

  async fn get_token(&self, client: &Client) -> Result<String> {
    let mut guard = self.cached_token.lock().await;
    // Refresh if missing or expiring within 5 minutes
    if guard
      .as_ref()
      .map(|t| t.expires_at < now_seconds() + 300)
      .unwrap_or(true)
    {
      let copilot_token = env::var(&self.config.api_key_env)?;
      let resp = client
        .get(TOKEN_URL)
        .header("User-Agent", USER_AGENT)
        .header("Authorization", format!("token {}", copilot_token))
        .header("X-GitHub-Api-Version", "2025-04-01")
        .send()
        .await?;
      if !resp.status().is_success() {
        return Err(anyhow!("Failed to fetch Copilot token: {}", resp.status()));
      }
      let token_resp: CopilotTokenResponse = resp.json().await?;
      *guard = Some(CachedToken {
        token: token_resp.token,
        expires_at: token_resp.expires_at,
      });
    }
    Ok(guard.as_ref().unwrap().token.clone())
  }
}

impl Infill for CopilotInfill {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String, language_id: String) -> Result<impl Iterator<Item = String>> {
    let token = self.get_token(&client).await?;
    let request_id = Uuid::new_v4().to_string();
    let prompt_tokens = prefix.split_whitespace().count();
    let suffix_tokens = suffix.split_whitespace().count();
    let response = client
      .post(COMPLETIONS_URL)
      .header("User-Agent", USER_AGENT)
      .header("Content-Type", "application/json")
      .header("Authorization", format!("Bearer {}", token))
      .header("X-Request-Id", &request_id)
      .header("X-GitHub-Api-Version", "2025-04-01")
      .header("Openai-Organization", "github-copilot")
      .header("VScode-SessionId", &self.session_id)
      .header("VScode-MachineId", &self.machine_id)
      .header("OpenAI-Intent", "copilot-ghost")
      .json(&CopilotCompletionsRequest {
        prompt: &prefix,
        suffix: &suffix,
        max_tokens: self.config.max_tokens,
        temperature: self.config.temperature.unwrap_or(0.0),
        top_p: self.config.top_p.unwrap_or(1.0),
        n: 1,
        stop: &self.config.stop,
        stream: true,
        extra: CopilotCompletionsExtra {
          language: &language_id,
          next_indent: 0,
          trim_by_indentation: true,
          prompt_tokens,
          suffix_tokens,
        },
      })
      .send()
      .await?;
    // Parse SSE stream: lines of "data: <json>" or "data: [DONE]"
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
          log::debug!("Copilot SSE chunk: {}", data);
          let chunk: CompletionsChunk = serde_json::from_str(data)?;
          for choice in chunk.choices {
            match completions.get_mut(0) {
              Some(s) => s.push_str(&choice.text),
              None => completions.push(choice.text),
            }
          }
        }
      }
    }
    Ok(completions.into_iter())
  }
}
