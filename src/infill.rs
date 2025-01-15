use std::{future::Future, iter, sync::Arc};

use anyhow::Result;
use either::Either;
use reqwest::Client;

pub trait Infill {
  fn infill(
    &self,
    client: Arc<Client>,
    prefix: String,
    suffix: String,
  ) -> impl Future<Output = Result<impl Iterator<Item = String>>> + Send;
}

impl<A: Infill + Sync, B: Infill + Sync> Infill for Either<A, B> {
  async fn infill(&self, client: Arc<Client>, prefix: String, suffix: String) -> Result<impl Iterator<Item = String>> {
    match self {
      Either::Left(a) => a.infill(client, prefix, suffix).await.map(Either::Left),
      Either::Right(b) => b.infill(client, prefix, suffix).await.map(Either::Right),
    }
  }
}

impl Infill for () {
  async fn infill(
    &self,
    _client: Arc<Client>,
    _prefix: String,
    _suffix: String,
  ) -> Result<impl Iterator<Item = String>> {
    Ok(iter::empty())
  }
}
