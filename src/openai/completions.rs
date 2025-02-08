use std::{env, sync::Arc};

use anyhow::Result;
use ramhorns::Content;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
  config::{ModelConfig, OpenAI, TemplateConfig},
  infill::Infill,
};

#[derive(Clone, PartialEq, Debug, Serialize)]
struct OpenAICompletionsRequest<'a> {
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

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct OpenAICompletionsChoice {
  text: String,
}

#[derive(Clone, PartialEq, Debug, Deserialize)]
struct OpenAICompletionsResponse {
  choices: Vec<OpenAICompletionsChoice>,
}

#[derive(Content)]
struct OpenAICompletionsContent {
  prefix: String,
  suffix: String,
}

impl Infill for (Arc<TemplateConfig>, Arc<ModelConfig<OpenAI>>) {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
    let request = client.post(&self.1.url);
    let request = if let Some(ref api_key_env) = self.1.api_key_env {
      request.bearer_auth(&env::var(api_key_env)?)
    } else {
      request
    };
    let response = request
      .json(&OpenAICompletionsRequest {
        model: &self.1.generation_config.model,
        prompt: self.0 .0.render(&OpenAICompletionsContent { prefix, suffix }),
        temperature: self.1.generation_config.temperature,
        top_p: self.1.generation_config.top_p,
        max_tokens: self.1.generation_config.max_tokens,
        stop: &self.1.generation_config.stop,
        seed: self.1.generation_config.seed,
      })
      .send()
      .await?
      .json::<OpenAICompletionsResponse>()
      .await?;

    Ok(response.choices.into_iter().map(|choice| choice.text))
  }
}
