mod config;
mod copilot;
mod infill;
mod llama_cpp;
mod mistral;
mod ollama;
mod openai;

use std::{env, fs::File, io::BufReader, sync::Arc};

use anyhow::{anyhow, Result};
use clap::Command;
use config::Config;
use crossbeam_channel::Sender;
use dashmap::DashMap;
use infill::Infill;
use lsp_server::{Connection, ErrorCode, Message, RequestId, Response as LspResponse};
use lsp_types::{
  notification::{Cancel, DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Exit, Notification},
  request::{InlineCompletionRequest, Request},
  CancelParams, DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, InitializeParams,
  InlineCompletionItem, InlineCompletionParams, InlineCompletionResponse, NumberOrString, OneOf, Range,
  ServerCapabilities, TextDocumentSyncKind, Uri,
};

use reqwest::Client;
use ropey::Rope;
use tokio::task::JoinHandle;

#[derive(Debug)]
struct Document {
  rope: Rope,
  version: i32,
  language_id: String,
}

#[derive(Debug)]
struct State {
  sender: Arc<Sender<Message>>,
  client: Arc<Client>,
  config: Config,
  documents: Arc<DashMap<Uri, Document>>,
  tasks: Arc<DashMap<RequestId, JoinHandle<Result<()>>>>,
}

impl State {
  fn inline_completion_request(&self, request_id: RequestId, params: InlineCompletionParams) -> Result<()> {
    let document = self
      .documents
      .get(&params.text_document_position.text_document.uri)
      .ok_or_else(|| {
        anyhow!(
          "Missing document: {}",
          params.text_document_position.text_document.uri.as_str()
        )
      })?;

    let index = document
      .rope
      .line_to_char(params.text_document_position.position.line as usize)
      + params.text_document_position.position.character as usize;
    let prefix = document.rope.slice(..index).to_string();
    let suffix = document.rope.slice(index..).to_string();
    let language_id = document.language_id.clone();

    let infill = self.config.get_infill();
    let client = self.client.clone();
    let sender = self.sender.clone();
    let tasks = self.tasks.clone();
    let request_id_c = request_id.clone();
    let future = async move {
      let completions = infill.infill(client, prefix, suffix, language_id).await;
      match completions {
        Result::Ok(completions) => {
          tasks.remove(&request_id_c);
          let range = Range::new(
            params.text_document_position.position,
            params.text_document_position.position,
          );
          let completion_items = completions
            .filter(|completion| !completion.is_empty())
            .map(|completion| InlineCompletionItem {
              insert_text: completion,
              filter_text: None,
              range: Some(range),
              command: None,
              insert_text_format: None,
            })
            .collect();
          sender.send(Message::Response(LspResponse::new_ok(
            request_id_c,
            InlineCompletionResponse::Array(completion_items),
          )))?;
        }
        Result::Err(error) => {
          tasks.remove(&request_id_c);
          sender.send(Message::Response(LspResponse::new_err(
            request_id_c,
            ErrorCode::RequestFailed as i32,
            format!("Failed to get response: {}", error),
          )))?;
        }
      }
      Ok(())
    };
    let handle = tokio::task::spawn(future);
    self.tasks.insert(request_id, handle);
    Ok(())
  }

  fn did_open_text_document(&mut self, params: DidOpenTextDocumentParams) -> Result<()> {
    let file = File::open(params.text_document.uri.path().as_str())?;
    let reader = BufReader::new(file);
    let rope = Rope::from_reader(reader)?;
    self.documents.insert(
      params.text_document.uri,
      Document {
        rope,
        version: params.text_document.version,
        language_id: params.text_document.language_id,
      },
    );
    Ok(())
  }

  fn did_close_text_document(&mut self, params: DidCloseTextDocumentParams) {
    self.documents.remove(&params.text_document.uri);
  }

  fn did_change_text_document(&mut self, params: DidChangeTextDocumentParams) -> Result<()> {
    for change in params.content_changes {
      if let Some(range) = change.range {
        let mut document = self
          .documents
          .get_mut(&params.text_document.uri)
          .ok_or_else(|| anyhow!("Missing document: {}", params.text_document.uri.as_str()))?;
        let start_index = document.rope.line_to_char(range.start.line as usize) + range.start.character as usize;
        let end_index = document.rope.line_to_char(range.end.line as usize) + range.end.character as usize;
        document.rope.remove(start_index..end_index);
        document.rope.insert(start_index, &change.text);
        document.version = params.text_document.version;
      } else {
        let rope = Rope::from_str(&change.text);
        let language_id = self
          .documents
          .get(&params.text_document.uri)
          .map(|d| d.language_id.clone())
          .unwrap_or_default();
        self.documents.insert(
          params.text_document.uri.clone(),
          Document {
            rope,
            version: params.text_document.version,
            language_id,
          },
        );
      }
    }
    Ok(())
  }

  fn cancel(&self, params: CancelParams) {
    let id: RequestId = match params.id {
      NumberOrString::Number(id) => id.into(),
      NumberOrString::String(id) => id.into(),
    };
    if let Some((_, handle)) = self.tasks.remove(&id) {
      handle.abort();
    }
  }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let _matches = Command::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .get_matches();

  let (connection, io_threads) = Connection::stdio();
  let server_capabilities = ServerCapabilities {
    execute_command_provider: None,
    inline_completion_provider: Some(OneOf::Left(true)),
    text_document_sync: Some(lsp_types::TextDocumentSyncCapability::Kind(
      TextDocumentSyncKind::INCREMENTAL,
    )),
    ..Default::default()
  };
  let initialize_params = connection.initialize(serde_json::to_value(server_capabilities)?)?;
  let initialize_params = serde_json::from_value::<InitializeParams>(initialize_params)?;

  let initialization_options = initialize_params
    .initialization_options
    .ok_or_else(|| anyhow!("Missing initialization options"))?;
  let config = serde_json::from_value::<Config>(initialization_options)?;

  let mut state = State {
    sender: Arc::new(connection.sender),
    client: Arc::new(reqwest::Client::new()),
    config,
    documents: Default::default(),
    tasks: Default::default(),
  };

  for msg in &connection.receiver {
    match msg {
      Message::Request(request) => {
        if request.method == InlineCompletionRequest::METHOD {
          let (request_id, params) = request.extract::<InlineCompletionParams>(InlineCompletionRequest::METHOD)?;
          state.inline_completion_request(request_id, params)?;
        }
      }
      Message::Notification(notification) => {
        if notification.method == DidOpenTextDocument::METHOD {
          let params: DidOpenTextDocumentParams = serde_json::from_value(notification.params)?;
          state.did_open_text_document(params)?;
        } else if notification.method == DidCloseTextDocument::METHOD {
          let params: DidCloseTextDocumentParams = serde_json::from_value(notification.params)?;
          state.did_close_text_document(params);
        } else if notification.method == DidChangeTextDocument::METHOD {
          let params: DidChangeTextDocumentParams = serde_json::from_value(notification.params)?;
          state.did_change_text_document(params)?;
        } else if notification.method == Cancel::METHOD {
          let params: CancelParams = serde_json::from_value(notification.params)?;
          state.cancel(params);
        } else if notification.method == Exit::METHOD {
          return Ok(());
        }
      }
      _ => {}
    }
  }

  io_threads.join().map_err(|e| e.into())
}
