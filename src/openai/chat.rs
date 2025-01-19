use std::{env, sync::Arc};

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
  chat::Chat,
  config::{ModelConfig, OpenAI},
};

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
struct OpenAIChatMessage {
  role: String,
  content: String,
}

#[derive(Clone, PartialEq, Debug, Serialize)]
struct OpanAIChatRequest<'a> {
  model: &'a Option<String>,
  messages: Vec<OpenAIChatMessage>,
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

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct OpenAIChatChoice {
  message: OpenAIChatMessage,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct OpenAIChatResponse {
  choices: Vec<OpenAIChatChoice>,
}

impl Chat for ModelConfig<OpenAI> {
  async fn chat(&self, client: Arc<Client>, messages: Vec<(String, String)>) -> Result<impl Iterator<Item = String>> {
    let request = client.post(&self.url);
    let request = if let Some(ref api_key_env) = self.api_key_env {
      request.bearer_auth(&env::var(api_key_env)?)
    } else {
      request
    };
    let response = request
      .json(&OpanAIChatRequest {
        model: &self.generation_config.model,
        messages: messages
          .into_iter()
          .map(|(role, content)| OpenAIChatMessage { role, content })
          .collect(),
        temperature: self.generation_config.temperature,
        top_p: self.generation_config.top_p,
        max_tokens: self.generation_config.max_tokens,
        stop: &self.generation_config.stop,
        seed: self.generation_config.seed,
      })
      .send()
      .await?
      .json::<OpenAIChatResponse>()
      .await?;

    Ok(response.choices.into_iter().map(|choice| choice.message.content))
  }
}
