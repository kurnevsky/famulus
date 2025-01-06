use std::sync::Arc;

use anyhow::Result;
use reqwest::Client;

pub trait Infill {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>>;
}
